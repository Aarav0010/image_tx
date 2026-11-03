#!/usr/bin/env python3
"""
image_link.py - send an IMAGE over the BPSK/QPSK link and rebuild it at the receiver.

How the image is sent
---------------------
1. The image is resized (default 128 x 128, colour) and its raw pixel bytes are turned into bits
   (8 bits per colour value). Raw pixels are used on purpose (not the JPEG file): a wrong bit then
   only changes one pixel, so noise shows up as visible "speckles" instead of destroying the file.
2. The bits are cut into frames. Every frame is
       [16 guard | 127-symbol BPSK preamble | 64-symbol BPSK header | 1024 data symbols]
   The header carries the frame number (16 bits, sent 4 times) so the receiver knows where in the
   image the frame belongs. BPSK frame = 1024 image bits, QPSK frame = 2048 image bits.
3. All frames are sent one after another, and the whole image repeats continuously.

Receiver
--------
Same receiver as psk_common.receive (matched filter, preamble sync, fine timing, frequency/phase
correction, phase tracking) plus header decoding. One full image cycle is decoded; frames that are
not detected are shown grey. Bits are compared with the original image bits -> BER, and pixels with
the original image -> PSNR.
"""

import math
import time
import numpy as np
from PIL import Image
import psk_common as pc

try:                       # GNU Radio is only needed to transmit/receive, not to analyse saved captures
    from gnuradio import gr, blocks, filter as gr_filter, channels
    HAS_GR = True
except ImportError:
    HAS_GR = False

try:
    from gnuradio import uhd
    HAS_UHD = True
except ImportError:
    HAS_UHD = False

WIN = 1500                 # samples per window for robust power measurements (0.1 ms at 15 MS/s)


def window_powers(x, n=WIN):
    """Mean power of consecutive short windows."""
    x = np.asarray(x)
    nw = max(len(x) // n, 1)
    return np.mean(np.abs(x[: nw * n].reshape(nw, -1)) ** 2, axis=1)


def robust_power(x):
    """Power level ignoring short bursts: median of the 0.1 ms window powers, scaled to the mean of
    noise-like windows (for Gaussian noise averaged over 1500 samples median ~= mean)."""
    return float(np.median(window_powers(x)))


def burst_fraction(x, samp_rate, signal_bw):
    """Fraction of 0.1 ms windows where the band OUTSIDE our signal (between the signal edge and the
    sampling edge) is > 6 dB above its usual level -> wideband interference such as WiFi packets."""
    x = np.asarray(x)
    nw = len(x) // WIN
    if nw < 4:
        return 0.0, 0.0
    X = np.fft.fftshift(np.fft.fft(x[: nw * WIN].reshape(nw, WIN), axis=1), axes=1)
    f = np.fft.fftshift(np.fft.fftfreq(WIN, 1.0 / samp_rate))
    oob = (np.abs(f) > 0.56 * signal_bw) & (np.abs(f) < 0.48 * samp_rate)
    if not np.any(oob):
        return 0.0, 0.0
    po = np.mean(np.abs(X[:, oob]) ** 2, axis=1)
    base = np.median(po)
    return float(np.mean(po > 4 * base)), float(10 * np.log10(po.max() / base))

N_HDR = 64          # header symbols (BPSK): 16-bit frame number x 4 repetitions
HDR_BITS = 16
HDR_REP = N_HDR // HDR_BITS


# Bandwidth settings.  occupied bandwidth = symbol rate x (1 + roll-off)
#   "10"  : 15 MS/s, 2 samples/symbol -> 7.5 Msym/s, roll-off 0.33 -> 7.5 x 1.33 = 9.98 MHz  (default)
#   "3.4" : 10 MS/s, 4 samples/symbol -> 2.5 Msym/s, roll-off 0.35 -> 2.5 x 1.35 = 3.375 MHz (old setting)
BW_PRESETS = {
    "10": {"samp_rate": 15e6, "sps": 2, "rolloff": 0.33},
    "3.4": {"samp_rate": 10e6, "sps": 4, "rolloff": 0.35},
}


def _hdr_symbols(idx):
    bits = np.array([(idx >> (HDR_BITS - 1 - i)) & 1 for i in range(HDR_BITS)])
    return (1.0 - 2.0 * np.tile(bits, HDR_REP)).astype(complex)


class ImageFrames:
    """Turns an image into a repeating stream of frames (and keeps everything the RX needs)."""

    def __init__(self, image_path, mod="bpsk", size=128, gray=False, bw="10"):
        self.mod = mod
        cfg = BW_PRESETS[str(bw)]
        self.bw = str(bw)
        self.samp_rate = cfg["samp_rate"]
        self.sps = cfg["sps"]
        self.rolloff = cfg["rolloff"]
        self.symbol_rate = self.samp_rate / self.sps
        self.occupied_bw = self.symbol_rate * (1 + self.rolloff)
        im = Image.open(image_path).convert("L" if gray else "RGB")
        if size:
            im = im.resize((size, size), Image.LANCZOS)
        self.pixels = np.array(im, dtype=np.uint8)
        self.shape = self.pixels.shape
        self.bits = np.unpackbits(self.pixels.reshape(-1))
        self.bits_per_frame = pc.N_PAYLOAD * pc.bits_per_symbol(mod)
        self.n_frames = int(math.ceil(len(self.bits) / self.bits_per_frame))
        pad = self.n_frames * self.bits_per_frame - len(self.bits)
        rng = np.random.default_rng(7)
        self.frame_bits = np.concatenate([self.bits, rng.integers(0, 2, pad)]).reshape(self.n_frames, -1)
        self.preamble = pc.m_sequence_127().astype(complex)
        self.taps = pc.rrc_taps(sps=self.sps, rolloff=self.rolloff)
        frames = []
        for k in range(self.n_frames):
            frames.append(np.concatenate([np.zeros(pc.N_GUARD, complex), self.preamble, _hdr_symbols(k),
                                          pc.map_bits(self.frame_bits[k], mod)]))
        self.symbols = np.concatenate(frames)
        self.frame_symbols = pc.N_GUARD + pc.N_PREAMBLE + N_HDR + pc.N_PAYLOAD
        self.frame_samples = self.frame_symbols * self.sps
        self.cycle_samples = self.n_frames * self.frame_samples
        # TX scaling: peak of the shaped signal = pc.TX_PEAK
        up = np.zeros(min(len(self.symbols), 40 * self.frame_symbols) * self.sps, complex)
        up[::self.sps] = self.symbols[: len(up) // self.sps]
        shaped = np.convolve(up, self.taps)
        self.tx_gain_lin = pc.TX_PEAK / np.max(np.abs(shaped))
        self.es_per_symbol = np.mean(np.abs(shaped * self.tx_gain_lin) ** 2) * self.sps * \
            self.frame_symbols / (self.frame_symbols - pc.N_GUARD)

    @property
    def n_image_bits(self):
        return len(self.bits)


def _frac_sample(x, pos, half=8):
    """Value of x at fractional positions pos (windowed-sinc interpolation, 2*half taps)."""
    n0 = np.floor(pos).astype(int)
    f = pos - n0
    k = np.arange(-half + 1, half + 1)
    idx = np.clip(n0[..., None] + k, 0, len(x) - 1)
    d = f[..., None] - k
    w = np.sinc(d) * (0.5 + 0.5 * np.cos(np.pi * d / half))
    return np.sum(x[idx] * w, axis=-1)


def receive_image(samples, tx, track_block=256):
    """Decode one full image cycle from captured samples. Returns a dict (see keys at the end)."""
    x = np.asarray(samples, dtype=complex)
    out = {"ok": False}
    F = tx.frame_samples
    if len(x) < tx.cycle_samples + 2 * F:
        out["error"] = "capture too short"
        return out
    rx_power_dbfs = 10 * np.log10(np.mean(np.abs(x) ** 2) + 1e-20)
    peak = float(np.max(np.maximum(np.abs(x.real), np.abs(x.imag))))

    mf = np.convolve(x, tx.taps, mode="same")
    pre_up = np.zeros(pc.N_PREAMBLE * tx.sps - (tx.sps - 1), complex)
    pre_up[::tx.sps] = tx.preamble
    corr = np.abs(pc._fft_xcorr(mf, pre_up))

    # --- frame timing: all frames are exactly F samples apart -> fold the correlation over F
    #     (robust even when single peaks are buried in noise)
    nfold = len(corr) // F
    folded = np.sum(corr[: nfold * F].reshape(nfold, F), axis=0)
    phase0 = int(np.argmax(folded))
    peak_ratio = folded[phase0] / (np.median(folded) + 1e-20)
    if peak_ratio < 3.0:
        out["error"] = "no frame lock (signal below noise)"
        out["rx_power_dbfs"] = float(rx_power_dbfs)
        return out
    need = (pc.N_PREAMBLE + N_HDR + pc.N_PAYLOAD) * tx.sps
    starts = np.array([phase0 + m * F for m in range(nfold + 1) if phase0 + m * F + need <= len(mf)])

    # --- re-sync every frame when the signal is strong enough. If the USB link drops a few samples
    #     (UHD prints "O"), the later frames move; following each frame's own preamble peak fixes it.
    typ_peak = float(np.median(corr[starts]))
    # largest noise peak inside one frame window (away from that window's main peak), typical over frames
    sub = starts[:: max(1, len(starts) // 60)]
    nm = []
    for c in sub:
        seg = corr[max(c - F // 2, 0): c + F // 2].copy()
        mid = int(np.argmax(seg))
        seg[max(mid - 4, 0): mid + 5] = 0
        nm.append(seg.max())
    noise_max = float(np.median(nm))
    strong = typ_peak > 2.0 * noise_max
    n_resync = 0
    if strong:
        W = F // 2
        new, off, s_ = [], 0, phase0
        while s_ + off + need <= len(mf):
            c = s_ + off
            lo, hi = max(c - W, 0), min(c + W + 1, len(corr))
            j = lo + int(np.argmax(corr[lo:hi]))
            if corr[j] > max(0.6 * typ_peak, 1.3 * noise_max) and abs(j - c) > 1:
                off += j - c
                n_resync += 1
                c = j
            if c + need <= len(mf):
                new.append(c)
            s_ += F
        starts = np.array(new)

    # --- fine (fractional-sample) timing: try sub-sample shifts of the sampling instant and keep
    #     the one where the preamble correlation is strongest (same idea as psk_common.receive)
    kk = tx.sps * np.arange(pc.N_PREAMBLE)

    def _metric(mu):
        Pm = _frac_sample(mf, starts[:, None] + kk[None, :] + mu)
        c = Pm @ np.conj(tx.preamble) / pc.N_PREAMBLE
        return np.sum(np.abs(c) ** 2) / np.sum(np.mean(np.abs(Pm) ** 2, axis=1))

    best_mu, best_v = 0.0, -1.0
    for mu in np.arange(-0.5, 0.5001, 0.125):
        v = _metric(mu)
        if v > best_v:
            best_mu, best_v = mu, v
    for mu in best_mu + np.arange(-0.1, 0.1001, 0.025):
        v = _metric(mu)
        if v > best_v:
            best_mu, best_v = mu, v

    nsym = pc.N_PREAMBLE + N_HDR + pc.N_PAYLOAD
    Y = _frac_sample(mf, starts[:, None] + tx.sps * np.arange(nsym)[None, :] + best_mu)
    P, H, D = Y[:, :pc.N_PREAMBLE], Y[:, pc.N_PREAMBLE:pc.N_PREAMBLE + N_HDR], Y[:, pc.N_PREAMBLE + N_HDR:]

    # --- frequency offset (one value for the whole capture) and per-frame channel
    z = P * np.conj(tx.preamble)[None, :]
    k_pre = np.arange(pc.N_PREAMBLE)
    # lag products do not depend on the (unknown) phase of each frame, so they can be summed over
    # ALL frames -> very robust even at negative SNR. Coarse (lag 1) then fine (lag 16, 63).
    # Every frame gets the same weight (normalised by its own power) and frames whose preamble power is
    # far above normal (hit by an interference burst) are left out - otherwise one strong WiFi burst
    # would dominate the sum and give a wrong frequency offset for the whole image.
    pf = np.mean(np.abs(P) ** 2, axis=1)
    keep = pf < 2.0 * np.median(pf)
    zn = z[keep] / pf[keep, None]
    w = float(np.angle(np.sum(zn[:, 1:] * np.conj(zn[:, :-1]))))
    for lag in (16, pc.N_PREAMBLE // 2):
        zr = zn * np.exp(-1j * w * k_pre)[None, :]
        w += float(np.angle(np.sum(zr[:, lag:] * np.conj(zr[:, :-lag])))) / lag
    w = np.full(len(starts), w)
    h = np.mean(z * np.exp(-1j * w[:, None] * k_pre[None, :]), axis=1)

    # --- header -> frame number (repetition code, then "frames are consecutive" consistency)
    k_hdr = np.arange(pc.N_PREAMBLE, pc.N_PREAMBLE + N_HDR)
    He = (H * np.exp(-1j * w[:, None] * k_hdr[None, :]) / h[:, None]).real
    soft = He.reshape(len(starts), HDR_REP, HDR_BITS).sum(axis=1)
    hbits = (soft < 0).astype(int)
    dec_idx = hbits @ (1 << np.arange(HDR_BITS - 1, -1, -1))
    pos = np.round((starts - starts[0]) / F).astype(int)
    offs = (dec_idx - pos) % tx.n_frames
    vals, cnts = np.unique(offs, return_counts=True)
    s0 = int(vals[np.argmax(cnts)])
    idx = (s0 + pos) % tx.n_frames
    header_ok = float(np.mean(dec_idx == idx))
    hdr_consistent = float(np.mean((np.diff(dec_idx) % tx.n_frames) == 1)) if len(dec_idx) > 1 else 0.0
    if strong and hdr_consistent > 0.9:
        # clean headers: after a sample drop the position count can be off by one, so trust a decoded
        # frame number when BOTH neighbours confirm it (n-1, n, n+1)
        N = tx.n_frames
        use = np.zeros(len(idx), bool)
        use[1:-1] = ((dec_idx[1:-1] - dec_idx[:-2]) % N == 1) & ((dec_idx[2:] - dec_idx[1:-1]) % N == 1) \
            & (dec_idx[1:-1] < N)
        idx = np.where(use, dec_idx, idx)

    # --- payload: equalise, track phase, decide
    k_dat = np.arange(pc.N_PREAMBLE + N_HDR, nsym)
    E = D * np.exp(-1j * w[:, None] * k_dat[None, :]) / h[:, None]
    # residual phase per block of 256 symbols (M-th power method). The preamble already fixed the
    # phase, so the correction is small: no unwrapping (avoids cycle slips at low SNR), and it is
    # only applied when the block estimate is reliable.
    M = 2 if tx.mod == "bpsk" else 4
    ref_m = 1.0 if tx.mod == "bpsk" else -1.0
    nb = pc.N_PAYLOAD // track_block
    Eb = E[:, : nb * track_block].reshape(len(starts), nb, track_block)
    acc = np.sum(Eb ** M, axis=2) * np.conj(ref_m)
    coh = np.abs(acc) / (np.sum(np.abs(Eb) ** M, axis=2) + 1e-20)
    est = np.where(coh > 0.25, np.angle(acc) / M, 0.0)
    phi = np.repeat(est, track_block, axis=1)
    E = E * np.exp(-1j * phi)
    bits_hat = pc.demap_symbols(E, tx.mod).reshape(len(starts), -1)

    # --- one full image cycle: first occurrence of every frame number
    rx_bits = np.full((tx.n_frames, tx.bits_per_frame), -1, dtype=int)
    used = []
    for i, k in enumerate(idx):
        if rx_bits[k, 0] < 0:
            rx_bits[k] = bits_hat[i]
            used.append(i)
    got = rx_bits[:, 0] >= 0
    flat_rx = rx_bits.reshape(-1)[: tx.n_image_bits]
    flat_tx = tx.bits
    valid = flat_rx >= 0
    errors = int(np.sum(flat_rx[valid] != flat_tx[valid]))
    n_bits = int(np.sum(valid))

    # SNR from EVM against the known transmitted symbols (frames actually used)
    ref_syms = np.stack([pc.map_bits(tx.frame_bits[k], tx.mod) for k in idx[used]])
    frame_evm2 = np.mean(np.abs(E[used] - ref_syms) ** 2, axis=1) / np.mean(np.abs(ref_syms) ** 2)
    evm2 = float(np.mean(frame_evm2))
    # frames hit by a short burst (interference / sample drop): error power > 3x the typical frame
    bad_frames = int(np.sum(frame_evm2 > 3.0 * np.median(frame_evm2)))
    frame_ber = np.mean(bits_hat[used] != tx.frame_bits[idx[used]], axis=1)
    good = frame_evm2 <= 3.0 * np.median(frame_evm2)
    ber_clean = float(np.mean(frame_ber[good])) if np.any(good) else float("nan")
    # SNR of a typical frame (median) - not pulled down by a few burst-hit frames
    esn0_db = -10 * np.log10(float(np.median(frame_evm2)) + 1e-20)
    esn0_mean_db = -10 * np.log10(evm2 + 1e-20)

    # rebuild the image (missing frames -> mid grey)
    b = np.where(flat_rx >= 0, flat_rx, 0).astype(np.uint8)
    pix = np.packbits(b)[: tx.pixels.size].reshape(tx.shape)
    missing = np.repeat(~got, tx.bits_per_frame)[: tx.n_image_bits]
    miss_pix = np.packbits(missing.astype(np.uint8))[: tx.pixels.size] > 0  # any missing bit in byte
    pix = pix.copy()
    pix.reshape(-1)[miss_pix] = 128
    mse = np.mean((pix.astype(float) - tx.pixels.astype(float)) ** 2)
    psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")

    out.update({
        "ok": True,
        "image": pix,
        "frames_received": int(np.sum(got)),
        "frames_total": tx.n_frames,
        "header_ok": header_ok,
        "bad_frames": bad_frames,
        "ber_clean": ber_clean,
        "resyncs": n_resync,
        "frame_evm_pct": 100 * np.sqrt(frame_evm2),
        "frame_ber": frame_ber,
        "frame_index": idx[used],
        "bit_errors": errors,
        "n_bits": n_bits,
        "ber": errors / max(n_bits, 1),
        "esn0_db": float(esn0_db),
        "esn0_mean_db": float(esn0_mean_db),
        "ebn0_db": float(esn0_db - 10 * np.log10(pc.bits_per_symbol(tx.mod))),
        "evm_pct": float(100 * math.sqrt(evm2)),
        "psnr_db": float(psnr),
        "cfo_hz": float(np.median(w) / (2 * math.pi) * tx.samp_rate / tx.sps),
        "rx_power_dbfs": float(rx_power_dbfs),
        "rx_peak": peak,
        "clipping": peak > 0.95,
    })
    return out


class ImageLink(gr.top_block if HAS_GR else object):
    """GNU Radio flowgraph:
         Vector Source (image frames, repeating) -> RRC interpolating FIR (x sps) -> Multiply Const
           -> USRP Sink (TX1 RX1, 5.2 GHz)          [hardware]
           -> Channel Model (noise, freq offset)    [simulation]
         USRP Source (RX2) -> Copy (on only while capturing) -> Vector Sink
    In simulation the "TX gain" changes the signal amplitude by the same number of dB as the USRP
    gain would, and the noise stays fixed - just like the real radio."""

    def __init__(self, tx, hardware=False, freq=pc.CENTER_FREQ, samp_rate=None,
                 tx_gain=20.0, rx_gain=40.0, sim_esn0_at_ref=2.6, sim_ref_gain=30.0,
                 freq_offset_hz=300.0, device_args="type=b200"):
        gr.top_block.__init__(self, "Image link")
        samp_rate = samp_rate or tx.samp_rate
        self.tx = tx
        self.hardware = hardware
        self.on = True
        self.scale = 1.0
        self.last_drops = 0
        self.sim_ref_gain = sim_ref_gain
        self.src = blocks.vector_source_c(tx.symbols.tolist(), True)
        self.rrc = gr_filter.interp_fir_filter_ccf(tx.sps, tx.taps.tolist())
        self.amp = blocks.multiply_const_cc(complex(tx.tx_gain_lin))
        self.connect(self.src, self.rrc, self.amp)
        self.sink = blocks.vector_sink_c()
        if hardware:
            if not HAS_UHD:
                raise RuntimeError("gnuradio.uhd not available")
            self.usrp_tx = uhd.usrp_sink(device_args, uhd.stream_args(cpu_format="fc32", channels=[0]))
            self.usrp_tx.set_subdev_spec("A:A", 0)
            self.usrp_tx.set_samp_rate(samp_rate)
            self.usrp_tx.set_center_freq(freq, 0)
            self.usrp_tx.set_bandwidth(samp_rate, 0)
            self.usrp_tx.set_antenna("TX/RX", 0)
            self.usrp_tx.set_gain(tx_gain, 0)
            self.usrp_rx = uhd.usrp_source(device_args, uhd.stream_args(cpu_format="fc32", channels=[0]))
            self.usrp_rx.set_subdev_spec("A:B", 0)
            self.usrp_rx.set_samp_rate(samp_rate)
            self.usrp_rx.set_center_freq(freq, 0)
            self.usrp_rx.set_bandwidth(samp_rate, 0)
            self.usrp_rx.set_antenna("RX2", 0)
            self.usrp_rx.set_gain(rx_gain, 0)
            self.valve = blocks.copy(gr.sizeof_gr_complex)
            self.valve.set_enabled(False)          # drop samples while not capturing (saves RAM)
            self.connect(self.amp, self.usrp_tx)
            self.connect(self.usrp_rx, self.valve, self.sink)
        else:
            noise_v = float(np.sqrt(tx.es_per_symbol / 10 ** (sim_esn0_at_ref / 10)))
            self.chan = channels.channel_model(noise_voltage=noise_v, frequency_offset=freq_offset_hz / samp_rate,
                                               epsilon=1.0, taps=[complex(np.exp(0.7j))], noise_seed=0,
                                               block_tags=False)
            # "receiver gain": scale so the simulated noise floor sits near -60 dBFS like the real B210
            self.rx_scale = blocks.multiply_const_cc(complex(10 ** (-60 / 20) / noise_v))
            self.head = blocks.head(gr.sizeof_gr_complex, 1)
            self.connect(self.amp, self.chan, self.rx_scale, self.head, self.sink)
            self.set_tx_gain(tx_gain)

    def _update_amp(self):
        self.amp.set_k(complex(self.tx.tx_gain_lin * self.scale if self.on else 0.0))

    def set_frames(self, tx):
        """Swap the transmitted image/modulation (call while the graph is stopped)."""
        self.tx = tx
        self.src.set_data(tx.symbols.tolist())
        self._update_amp()

    def set_tx_on(self, on):
        self.on = bool(on)
        self._update_amp()

    def set_tx_gain(self, g):
        if self.hardware:
            self.usrp_tx.set_gain(float(g), 0)
        else:
            self.scale = 10 ** ((float(g) - self.sim_ref_gain) / 20)
            self._update_amp()

    def set_rx_gain(self, g):
        if self.hardware:
            self.usrp_rx.set_gain(float(g), 0)

    def capture(self, n, settle_s=0.3):
        if not self.hardware:
            self.head.set_length(n)
            self.head.reset()
            self.sink.reset()
            self.run()
            return np.array(self.sink.data(), dtype=complex)
        time.sleep(settle_s)
        self.sink.reset()
        self.valve.set_enabled(True)
        t0 = time.time()
        while len(self.sink.data()) < n and time.time() - t0 < 15.0:
            time.sleep(0.02)
        self.valve.set_enabled(False)
        data = np.array(self.sink.data(), dtype=complex)[:n]
        # UHD marks every gap in the stream (USB overflow "O") with a new rx_time tag
        try:
            import pmt
            self.last_drops = sum(1 for t in self.sink.tags() if pmt.symbol_to_string(t.key) == "rx_time")
        except Exception:
            self.last_drops = -1
        self.sink.reset()
        return data
