#!/usr/bin/env python3
"""
psk_common.py - shared frame definition and data-aided PSK receiver.

Used by both the simulation (run_sim.py) and the hardware run (run_hardware.py),
so the transmitter frame and the receiver are identical in both cases.

Frame (symbol level):
    [ 16 guard zeros | 127-symbol BPSK preamble (m-sequence) | 1024 payload symbols ]
The preamble is always BPSK. The payload is BPSK or QPSK (Gray mapped).

Receiver (numpy, data-aided):
    RRC matched filter -> preamble correlation (frame + timing) -> per-frame
    CFO and channel (gain/phase) estimate from the preamble -> equalise ->
    block-wise decision-directed phase tracking -> decisions -> BER / EVM / SNR.
"""

import math
import numpy as np

# ------------------------------------------------------------------------------
# Link parameters (sir's spec: 5.2 GHz, 10 MHz)
# ------------------------------------------------------------------------------
CENTER_FREQ = 5.2e9     # Hz
SAMP_RATE = 10e6        # samples/s  (USRP sample rate and analog bandwidth = 10 MHz)
SPS = 4                 # samples per symbol -> symbol rate 2.5 Msym/s
ROLLOFF = 0.35          # RRC excess bandwidth -> occupied BW = 2.5 * 1.35 = 3.375 MHz
RRC_SPAN = 11           # RRC filter span in symbols
N_GUARD = 16
N_PREAMBLE = 127
N_PAYLOAD = 1024
TX_PEAK = 0.5           # peak amplitude of the shaped TX signal (keeps DAC out of clipping)
PAYLOAD_SEED = 2301     # fixed seed so TX and RX agree on payload bits


def rrc_taps(sps=SPS, rolloff=ROLLOFF, span=RRC_SPAN):
    """Root-raised-cosine taps (unit energy). Same shape GNU Radio firdes produces."""
    n = np.arange(-span * sps // 2, span * sps // 2 + 1, dtype=float)
    t = n / sps
    taps = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            taps[i] = 1.0 - rolloff + 4 * rolloff / math.pi
        elif abs(abs(4 * rolloff * ti) - 1.0) < 1e-9:
            taps[i] = (rolloff / math.sqrt(2)) * (
                (1 + 2 / math.pi) * math.sin(math.pi / (4 * rolloff))
                + (1 - 2 / math.pi) * math.cos(math.pi / (4 * rolloff)))
        else:
            num = (math.sin(math.pi * ti * (1 - rolloff))
                   + 4 * rolloff * ti * math.cos(math.pi * ti * (1 + rolloff)))
            den = math.pi * ti * (1 - (4 * rolloff * ti) ** 2)
            taps[i] = num / den
    return taps / np.sqrt(np.sum(taps ** 2))


def m_sequence_127():
    """Length-127 maximal-length sequence (x^7 + x^6 + 1) mapped to +/-1."""
    reg = [1] * 7
    out = []
    for _ in range(127):
        out.append(reg[-1])
        fb = reg[6] ^ reg[5]
        reg = [fb] + reg[:-1]
    return 1.0 - 2.0 * np.array(out, dtype=float)


def bits_per_symbol(mod):
    return {"bpsk": 1, "qpsk": 2}[mod]


def map_bits(bits, mod):
    """Gray mapping. BPSK: 0->+1, 1->-1. QPSK: (b0,b1) -> ((1-2b0) + j(1-2b1))/sqrt2."""
    bits = np.asarray(bits, dtype=int)
    if mod == "bpsk":
        return (1.0 - 2.0 * bits).astype(complex)
    b = bits.reshape(-1, 2)
    return ((1.0 - 2.0 * b[:, 0]) + 1j * (1.0 - 2.0 * b[:, 1])) / math.sqrt(2)


def demap_symbols(sym, mod):
    if mod == "bpsk":
        return (sym.real < 0).astype(int)
    b0 = (sym.real < 0).astype(int)
    b1 = (sym.imag < 0).astype(int)
    return np.stack([b0, b1], axis=-1).reshape(*sym.shape[:-1], -1)


class Frame:
    """Holds everything the TX and the RX need to agree on."""

    def __init__(self, mod="bpsk"):
        assert mod in ("bpsk", "qpsk")
        self.mod = mod
        self.preamble = m_sequence_127().astype(complex)
        rng = np.random.default_rng(PAYLOAD_SEED)
        self.payload_bits = rng.integers(0, 2, N_PAYLOAD * bits_per_symbol(mod))
        self.payload_syms = map_bits(self.payload_bits, mod)
        self.symbols = np.concatenate([np.zeros(N_GUARD, complex), self.preamble, self.payload_syms])
        self.n_symbols = len(self.symbols)
        self.frame_samples = self.n_symbols * SPS
        self.taps = rrc_taps()
        # Gain so that the shaped (interpolated) signal peaks at TX_PEAK.
        shaped = self.shape(np.tile(self.symbols, 3), gain=1.0)
        self.tx_gain_lin = TX_PEAK / np.max(np.abs(shaped))
        shaped = shaped * self.tx_gain_lin
        mid = shaped[self.frame_samples: 2 * self.frame_samples]
        # Average energy per non-zero symbol at the TX output (used for sim SNR calibration).
        self.es_per_symbol = np.sum(np.abs(mid) ** 2) / (N_PREAMBLE + N_PAYLOAD)

    def shape(self, symbols, gain=None):
        """Upsample-by-SPS + RRC filter, exactly like GNU Radio interp_fir_filter_ccf."""
        up = np.zeros(len(symbols) * SPS, complex)
        up[::SPS] = symbols
        y = np.convolve(up, self.taps)
        return y * (self.tx_gain_lin if gain is None else gain)


def _fft_xcorr(x, ref):
    """c[n] = sum_k x[n+k] * conj(ref[k])  for n = 0..len(x)-len(ref)."""
    n = len(x) + len(ref) - 1
    nfft = 1 << (n - 1).bit_length()
    c = np.fft.ifft(np.fft.fft(x, nfft) * np.conj(np.fft.fft(ref, nfft)))
    return c[: len(x) - len(ref) + 1]


def receive(samples, frame, max_frames=None, track_block=256):
    """
    Data-aided receiver.  Returns a dict with BER, EVM, Es/N0, Eb/N0, the equalised
    payload symbols (for constellation plots) and some link diagnostics.
    """
    x = np.asarray(samples, dtype=complex)
    out = {"n_frames": 0, "ok": False}
    if len(x) < 3 * frame.frame_samples:
        out["error"] = "capture too short"
        return out

    rx_power_dbfs = 10 * np.log10(np.mean(np.abs(x) ** 2) + 1e-20)
    peak = float(np.max(np.maximum(np.abs(x.real), np.abs(x.imag))))

    # 1) RRC matched filter
    mf = np.convolve(x, frame.taps, mode="same")

    # 2) Preamble correlation at full rate (preamble upsampled with zeros)
    pre_up = np.zeros(N_PREAMBLE * SPS - (SPS - 1), complex)
    pre_up[::SPS] = frame.preamble
    corr = np.abs(_fft_xcorr(mf, pre_up))
    thr = 0.5 * np.max(corr)
    min_gap = int(0.8 * frame.frame_samples)

    # Greedy peak picking: strongest first, enforcing frame spacing
    order = np.argsort(corr)[::-1]
    starts = []
    taken = np.zeros(len(corr), bool)
    for idx in order:
        if corr[idx] < thr:
            break
        if taken[idx]:
            continue
        starts.append(idx)
        taken[max(0, idx - min_gap): idx + min_gap] = True
    starts = sorted(starts)

    need = (N_PREAMBLE + N_PAYLOAD) * SPS
    starts = [s for s in starts if s + need <= len(mf)]
    if max_frames:
        starts = starts[:max_frames]
    if not starts:
        out["error"] = "no frames detected"
        return out
    # Lock check: genuine frames repeat exactly every frame_samples. Noise peaks do not.
    if len(starts) >= 3:
        gaps = np.diff(starts)
        good = np.sum(np.abs(gaps - frame.frame_samples * np.round(gaps / frame.frame_samples)) <= 2)
        peak_ratio = np.mean(corr[starts]) / (np.median(corr) + 1e-20)
        if good < 0.6 * len(gaps) or peak_ratio < 6.0:
            out["error"] = "no frame lock (signal below noise)"
            return out
    else:
        out["error"] = "too few frames"
        return out

    # 2a) All frames share one timing (same radio clock): snap every start to the most common
    #     sample phase (mod frame length). Fixes +/-1 sample ties when the delay is ~half a sample.
    F = frame.frame_samples
    res = (np.array(starts) - starts[0]) % F
    vals, cnts = np.unique(res, return_counts=True)
    ref = vals[np.argmax(cnts)]
    adj = (res - ref + F // 2) % F - F // 2
    starts = [int(s - a) for s, a in zip(starts, adj)]
    starts = [s for s in starts if s >= 0 and s + need <= len(mf)]

    # 2b) Fine (fractional-sample) timing. The TX-RX delay is the same for every frame, so
    #     try fractional shifts of the matched-filter output in 1/8-sample steps (FFT phase ramp)
    #     and keep the one that maximises the preamble correlation over all frames.
    #     Without this the sampling point can be up to 1/8 symbol off -> ISI -> SNR ceiling ~19 dB.
    st = np.array(starts)
    kk = SPS * np.arange(N_PREAMBLE)
    MF = np.fft.fft(mf)
    fr = np.fft.fftfreq(len(mf))
    def _metric(m):
        # fraction of preamble energy explained by (gain x known preamble) = preamble SNR proxy
        Pm = m[st[:, None] + kk[None, :]]
        c = Pm @ np.conj(frame.preamble) / N_PREAMBLE
        return np.sum(np.abs(c) ** 2) / np.sum(np.mean(np.abs(Pm) ** 2, axis=1))

    def _shift(mu):
        return mf if mu == 0 else np.fft.ifft(MF * np.exp(2j * np.pi * fr * mu))

    best_mu, best_metric = 0.0, -1.0
    for mu in np.arange(-0.5, 0.5001, 0.125):            # coarse: 1/8 sample
        v = _metric(_shift(mu))
        if v > best_metric:
            best_mu, best_metric = mu, v
    for mu in best_mu + np.arange(-0.1, 0.1001, 0.025):   # fine: 1/40 sample
        v = _metric(_shift(mu))
        if v > best_metric:
            best_mu, best_metric = mu, v
    best_mf = _shift(best_mu)
    mf = best_mf
    out["timing_offset_samples"] = float(best_mu)

    # 3) Symbol-spaced samples of each frame -> matrix [frames x symbols]
    idx = np.array(starts)[:, None] + SPS * np.arange(N_PREAMBLE + N_PAYLOAD)[None, :]
    Y = mf[idx]
    P = Y[:, :N_PREAMBLE]
    D = Y[:, N_PREAMBLE:]

    # 4) Per-frame CFO (rad/symbol) and complex channel gain from the preamble
    #    coarse: lag-1 phase difference (wide range, noisy)
    #    fine  : lag-63 phase difference between preamble halves (accurate)
    z = P * np.conj(frame.preamble)[None, :]
    k_pre = np.arange(N_PREAMBLE)
    w = np.angle(np.sum(z[:, 1:] * np.conj(z[:, :-1]), axis=1))
    lag = N_PREAMBLE // 2
    for _ in range(2):
        zr = z * np.exp(-1j * w[:, None] * k_pre[None, :])
        w = w + np.angle(np.sum(zr[:, lag:2 * lag] * np.conj(zr[:, :lag]), axis=1)) / lag
    # The CFO is a property of the radios, not the frame: use one robust value for all frames.
    w = np.full_like(w, np.median(w))
    z_d = z * np.exp(-1j * w[:, None] * k_pre[None, :])
    h = np.mean(z_d, axis=1)
    noise_var_pre = np.mean(np.abs(P * np.exp(-1j * w[:, None] * k_pre[None, :])
                                   - h[:, None] * frame.preamble[None, :]) ** 2, axis=1)

    # 5) Equalise payload (derotate CFO, divide by channel)
    k_dat = np.arange(N_PREAMBLE, N_PREAMBLE + N_PAYLOAD)
    E = D * np.exp(-1j * w[:, None] * k_dat[None, :]) / h[:, None]

    # 6) Block-wise decision-directed phase tracking (M-th power, unwrapped)
    M = 2 if frame.mod == "bpsk" else 4
    ref_m = 1.0 if frame.mod == "bpsk" else -1.0      # (e^{j pi/4})^4 = -1
    nb = N_PAYLOAD // track_block
    Eb = E[:, : nb * track_block].reshape(len(starts), nb, track_block)
    est = np.angle(np.sum(Eb ** M, axis=2) * np.conj(ref_m))
    est = np.concatenate([np.zeros((len(starts), 1)), est], axis=1)
    phi = np.unwrap(est, axis=1)[:, 1:] / M
    phi_full = np.repeat(phi, track_block, axis=1)
    if phi_full.shape[1] < N_PAYLOAD:
        phi_full = np.concatenate(
            [phi_full, np.repeat(phi_full[:, -1:], N_PAYLOAD - phi_full.shape[1], axis=1)], axis=1)
    E = E * np.exp(-1j * phi_full)

    # 7) Decisions, BER, EVM-based SNR
    bits_hat = demap_symbols(E, frame.mod).reshape(len(starts), -1)
    errors = int(np.sum(bits_hat != frame.payload_bits[None, :]))
    n_bits = bits_hat.size
    err_vec = E - frame.payload_syms[None, :]
    evm2 = np.mean(np.abs(err_vec) ** 2) / np.mean(np.abs(frame.payload_syms) ** 2)
    esn0_db = -10 * np.log10(evm2 + 1e-20)
    ebn0_db = esn0_db - 10 * np.log10(bits_per_symbol(frame.mod))

    out.update({
        "ok": True,
        "n_frames": len(starts),
        "n_bits": n_bits,
        "bit_errors": errors,
        "ber": errors / n_bits,
        "evm_pct": 100 * math.sqrt(evm2),
        "esn0_db": esn0_db,
        "ebn0_db": ebn0_db,
        "esn0_preamble_db": float(10 * np.log10(np.mean(np.abs(h) ** 2) / np.mean(noise_var_pre))),
        "cfo_hz": float(np.median(w) / (2 * math.pi) * SAMP_RATE / SPS),
        "rx_power_dbfs": float(rx_power_dbfs),
        "rx_peak": peak,
        "clipping": peak > 0.95,
        "payload_eq": E,
    })
    return out


def theory_ber(ebn0_db):
    """BPSK and Gray-coded QPSK have the same bit error rate vs Eb/N0: Q(sqrt(2 Eb/N0))."""
    ebn0 = 10 ** (np.asarray(ebn0_db, float) / 10)
    return 0.5 * np.vectorize(math.erfc)(np.sqrt(ebn0))
