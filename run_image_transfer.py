#!/usr/bin/env python3
"""
run_image_transfer.py - send an image with BPSK and QPSK at 5.2 GHz and rebuild it at the receiver,
for a few USRP TX gains (default 20, 22, 24, 26, 28, 30 dB - low power only).

For every modulation and every TX gain it:
  1. turns the transmitter output OFF and records the noise level at the receiver (dBFS),
  2. turns it ON, records one full image transmission and measures received power,
  3. decodes the image, counts bit errors (BER), measures SNR (from EVM) and image quality (PSNR),
  4. saves the received image.

Examples
  python run_image_transfer.py --tag clear                 (real B210)
  python run_image_transfer.py --sim                       (no hardware, simulated noise)
  python run_image_transfer.py --gains 20,25,30 --rx-gain 50 --gray
"""
import argparse
import csv
import datetime
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import psk_common as pc
from image_link import ImageFrames, ImageLink, receive_image, robust_power, burst_fraction


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self.out.flush()
        self.f.flush()


def find_image(arg):
    cands = [arg] if arg else []
    cands += [os.path.join(HERE, "image.jpg"), os.path.join(HERE, "image.png"),
              os.path.join(HERE, "..", "image.jpg"), os.path.join(HERE, "..", "sample_assets", "image.jpg")]
    for c in cands:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    sys.exit("Image not found. Put image.jpg in this folder or use --image <file>.")


def db(x):
    return 10 * np.log10(max(x, 1e-20))


def save_png(arr, path, zoom=1):
    im = Image.fromarray(arr)
    if zoom > 1:
        im = im.resize((arr.shape[1] * zoom, arr.shape[0] * zoom), Image.NEAREST)
    im.save(path)



def print_header(m):
    print("\n--- %s (%d bits/symbol) ---" % (m.upper(), pc.bits_per_symbol(m)))
    print("%6s | %7s %7s %7s | %7s %7s | %7s %7s %8s %8s | %6s %4s %5s %5s" %
          ("TXgain", "noise", "rx pwr", "SNR(pw)", "Es/N0", "Eb/N0", "frames", "bit err", "BER", "BER*",
           "PSNR", "bad", "intf", "drops"))
    print("%6s | %7s %7s %7s | %7s %7s | %7s %7s %8s %8s | %6s %4s %5s %5s" %
          ("dB", "dBFS", "dBFS", "dB", "dB", "dB", "", "", "", "", "dB", "frm", "%", ""))


def analyse_point(m, g, tx, x, noise, drops, rx_gain, outdir):
    """Decode one capture, print one table line, save the received image. Returns (row, image, frame EVM)."""
    # noise / power: median of 0.1 ms windows, so short interference bursts do not change the level
    p_noise = robust_power(noise) if noise is not None and len(noise) else float("nan")
    p_rx = robust_power(x)
    snr_pw = db(max(p_rx - p_noise, 1e-20) / p_noise) if np.isfinite(p_noise) else float("nan")
    intf, intf_peak = burst_fraction(x, tx.samp_rate, tx.occupied_bw)
    r = receive_image(x, tx)
    row = {"mod": m, "tx_gain_db": g, "rx_gain_db": rx_gain,
           "noise_dbfs": round(db(p_noise), 2) if np.isfinite(p_noise) else "",
           "rx_power_dbfs": round(db(p_rx), 2), "snr_power_db": round(snr_pw, 2) if np.isfinite(snr_pw) else "",
           "interference_pct": round(100 * intf, 1), "interference_peak_db": round(intf_peak, 1),
           "usb_drops": drops}
    nz = row["noise_dbfs"] if row["noise_dbfs"] != "" else float("nan")
    sp = snr_pw
    if not r["ok"]:
        row.update({"frames_rx": 0, "frames_total": tx.n_frames, "error": r.get("error", "")})
        print("%6.0f | %7.1f %7.1f %7.1f | NOT DECODED: %s" % (g, nz, row["rx_power_dbfs"], sp, r.get("error", "")))
        return row, None, None
    row.update({"esn0_db": round(r["esn0_db"], 2), "ebn0_db": round(r["ebn0_db"], 2),
                "evm_pct": round(r["evm_pct"], 1),
                "frames_rx": r["frames_received"], "frames_total": r["frames_total"],
                "bits": r["n_bits"], "bit_errors": r["bit_errors"], "ber": r["ber"],
                "ber_clean": r["ber_clean"], "theory_ber": float(pc.theory_ber(r["ebn0_db"])),
                "psnr_db": round(r["psnr_db"], 2) if np.isfinite(r["psnr_db"]) else 99.0,
                "bad_frames": r["bad_frames"], "resyncs": r["resyncs"],
                "esn0_mean_db": round(r["esn0_mean_db"], 2),
                "cfo_hz": round(r["cfo_hz"], 1), "clipping": r["clipping"]})
    img = r["image"]
    save_png(img, os.path.join(outdir, "rx_%s_tx%02d.png" % (m, int(round(g)))))
    save_png(img, os.path.join(outdir, "rx_%s_tx%02d_x4.png" % (m, int(round(g)))), zoom=4)
    print("%6.0f | %7.1f %7.1f %7.1f | %7.2f %7.2f | %3d/%-3d %7d %8.1e %8.1e | %6.1f %4d %5.1f %5d%s" %
          (g, nz, row["rx_power_dbfs"], sp, r["esn0_db"], r["ebn0_db"], r["frames_received"], r["frames_total"],
           r["bit_errors"], r["ber"], r["ber_clean"], row["psnr_db"], r["bad_frames"], 100 * intf, drops,
           "  CLIPPING" if r["clipping"] else ""))
    return row, img, r["frame_evm_pct"]


def write_outputs(outdir, rows, images, frame_evm, gains, mods, frames, gray, sim, rx_gain):
    f0 = frames[mods[0]]
    class _A:  # small shim so the plotting code below keeps its names
        pass
    args = _A()
    args.gray, args.sim, args.rx_gain = gray, sim, rx_gain

    # ---------------- save table ----------------
    keys = ["mod", "tx_gain_db", "rx_gain_db", "noise_dbfs", "rx_power_dbfs", "snr_power_db", "esn0_db",
            "ebn0_db", "evm_pct", "frames_rx", "frames_total", "bits", "bit_errors", "ber", "theory_ber",
            "psnr_db", "bad_frames", "ber_clean", "interference_pct", "interference_peak_db", "usb_drops", "resyncs", "esn0_mean_db", "cfo_hz", "clipping", "error"]
    with open(os.path.join(outdir, "image_results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in keys})

    # ---------------- figure 1: image grid ----------------
    nc = len(gains) + 1
    fig, axs = plt.subplots(len(mods), nc, figsize=(2.1 * nc, 2.5 * len(mods) + 0.4), squeeze=False)
    cmap = "gray" if args.gray else None
    for i, m in enumerate(mods):
        axs[i, 0].imshow(f0.pixels, cmap=cmap, vmin=0, vmax=255, interpolation="nearest")
        axs[i, 0].set_title("Sent image", fontsize=9)
        axs[i, 0].set_ylabel(m.upper(), fontsize=12, fontweight="bold")
        for j, g in enumerate(gains):
            ax = axs[i, j + 1]
            row = next(rw for rw in rows if rw["mod"] == m and rw["tx_gain_db"] == g)
            img = images[(m, g)]
            if img is not None:
                ax.imshow(img, cmap=cmap, vmin=0, vmax=255, interpolation="nearest")
                ax.set_title("TX %g dB\nEb/N0 %.1f dB\nBER %.1e" % (g, row["ebn0_db"], row["ber"]), fontsize=8)
            else:
                ax.text(0.5, 0.5, "not\ndecoded", ha="center", va="center", fontsize=10)
                ax.set_title("TX %g dB" % g, fontsize=8)
        for ax in axs[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Image received over the %s at 5.2 GHz, %.1f MHz bandwidth (RX gain %.0f dB)" %
                 ("simulated link" if args.sim else "air (USRP B210)", f0.occupied_bw / 1e6, args.rx_gain),
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_img1_received_images.png"), dpi=150)
    plt.close(fig)

    # ---------------- figure 2: noise / power / SNR vs TX gain ----------------
    def _v(rw, k):
        v = rw.get(k, "")
        return np.nan if v == "" else float(v)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for m, mk in zip(mods, ["o", "s"]):
        rr = [rw for rw in rows if rw["mod"] == m]
        g = [rw["tx_gain_db"] for rw in rr]
        axs[0].plot(g, [_v(rw, "rx_power_dbfs") for rw in rr], mk + "-", label="%s received power" % m.upper())
        axs[0].plot(g, [_v(rw, "noise_dbfs") for rw in rr], mk + "--", label="%s noise (TX off)" % m.upper())
        axs[1].plot(g, [_v(rw, "esn0_db") for rw in rr], mk + "-", label="%s Es/N0 (EVM)" % m.upper())
        axs[1].plot(g, [_v(rw, "snr_power_db") for rw in rr], mk + ":", label="%s SNR (power)" % m.upper())
        axs[2].plot(g, [_v(rw, "interference_pct") for rw in rr], mk + "-", label=m.upper())
    axs[0].set_xlabel("USRP TX gain (dB)")
    axs[0].set_ylabel("level at receiver (dBFS)")
    axs[0].set_title("Received power and noise level")
    axs[1].set_xlabel("USRP TX gain (dB)")
    axs[1].set_ylabel("SNR (dB)")
    axs[1].set_title("SNR at the receiver")
    axs[2].set_xlabel("USRP TX gain (dB)")
    axs[2].set_ylabel("% of time")
    axs[2].set_title("Wideband interference (e.g. WiFi) during capture")
    for ax in axs:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_img2_noise_snr_vs_txgain.png"), dpi=150)
    plt.close(fig)

    # ---------------- figure 3: BER and PSNR ----------------
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    eb = np.linspace(-6, 12, 200)
    axs[0].semilogy(eb, pc.theory_ber(eb), "k-", lw=1, label="theory BPSK/QPSK")
    for m, mk in zip(mods, ["o", "s"]):
        rr = [rw for rw in rows if rw["mod"] == m and "ebn0_db" in rw]
        if not rr:
            continue
        ber_plot = [max(rw["ber"], 0.5 / max(rw["bits"], 1)) for rw in rr]
        axs[0].semilogy([rw["ebn0_db"] for rw in rr], ber_plot, mk, ms=7, label="%s measured (all frames)" % m.upper())
        bc = [max(float(rw["ber_clean"]), 0.5 / max(rw["bits"], 1)) for rw in rr]
        axs[0].semilogy([rw["ebn0_db"] for rw in rr], bc, mk, ms=11, mfc="none",
                        label="%s without burst-hit frames" % m.upper())
        axs[1].plot([rw["tx_gain_db"] for rw in rr], [rw["psnr_db"] for rw in rr], mk + "-", label=m.upper())
    axs[0].set_ylim(1e-7, 1)
    axs[0].set_xlabel("measured Eb/N0 (dB)")
    axs[0].set_ylabel("bit error rate")
    axs[0].set_title("BER of the image bits (0 errors plotted at 1/(2N))")
    axs[1].set_xlabel("USRP TX gain (dB)")
    axs[1].set_ylabel("PSNR (dB)  (99 = perfect copy)")
    axs[1].set_title("Image quality at the receiver")
    for ax in axs:
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_img3_ber_psnr.png"), dpi=150)
    plt.close(fig)

    # ---------------- figure 4: quality of every frame (shows short bursts) ----------------
    fig, axs = plt.subplots(len(mods), 1, figsize=(11, 3.2 * len(mods)), squeeze=False)
    for i, m in enumerate(mods):
        ax = axs[i, 0]
        for g in gains:
            if (m, g) in frame_evm:
                fe = frame_evm[(m, g)]
                t_ms = np.arange(len(fe)) * frames[m].frame_samples / frames[m].samp_rate * 1e3
                ax.semilogy(t_ms, fe, lw=0.8, label="TX %g dB" % g)
        ax.set_title("%s: error (EVM) of every frame in the order received - spikes = short bursts "
                     "(interference / USB drop)" % m.upper(), fontsize=9)
        ax.set_xlabel("time in capture (ms)")
        ax.set_ylabel("frame EVM (%)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, ncol=6)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_img4_per_frame_quality.png"), dpi=150)
    plt.close(fig)

    print("\nnoise / rx pwr = median of 0.1 ms windows (short bursts ignored)")
    print("BER* = BER without burst-hit frames, bad = frames hit by a burst (error > 3x normal)")
    print("intf = % of time with wideband interference outside our 10 MHz band (e.g. WiFi), drops = USB gaps")
    print("\nSaved: %s" % outdir)
    print("  original.png, rx_<mod>_txXX.png (and _x4 enlarged), fig_img1_received_images.png,")
    print("  fig_img2_noise_snr_vs_txgain.png, fig_img3_ber_psnr.png, fig_img4_per_frame_quality.png,")
    print("  image_results.csv, image_run_log.txt, captures.npz (raw received samples)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="image file (default: image.jpg next to this script or in D:\\BTP)")
    ap.add_argument("--mod", default="both", choices=["bpsk", "qpsk", "both"])
    ap.add_argument("--gains", default="20,22,24,26,28,30", help="TX gains in dB (comma separated)")
    ap.add_argument("--rx-gain", type=float, default=40.0)
    ap.add_argument("--freq", type=float, default=pc.CENTER_FREQ)
    ap.add_argument("--bw", default="10", choices=["10", "3.4"],
                    help="occupied signal bandwidth in MHz: 10 (15 MS/s, 7.5 Msym/s, default) or 3.4 (old)")
    ap.add_argument("--size", type=int, default=128, help="image is resized to size x size (0 = keep)")
    ap.add_argument("--gray", action="store_true", help="send grayscale (3x fewer bits)")
    ap.add_argument("--sim", action="store_true", help="simulation instead of the B210")
    ap.add_argument("--sim-esn0-at-30", type=float, default=11.5,
                    help="sim only: Es/N0 (dB) at TX gain 30 for the 10 MHz signal. 11.5 dB = measured on the "
                         "B210 (clear link, RX gain 40, 1 Nov 2026). For --bw 3.4 it is raised by 4.8 dB.")
    ap.add_argument("--tag", default="", help="condition name for the folder (clear, hand, plate...)")
    args = ap.parse_args()

    gains = [float(g) for g in args.gains.split(",") if g.strip()]
    mods = ["bpsk", "qpsk"] if args.mod == "both" else [args.mod]
    img_path = find_image(args.image)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    tag = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in args.tag.strip())
    name = "results_image_%s%s_%sMHz%s" % ("sim_" if args.sim else "", stamp, args.bw.replace(".", "p"),
                                           ("_" + tag) if tag else "")
    outdir = os.path.join(HERE, name)
    os.makedirs(outdir, exist_ok=True)
    sys.stdout = Tee(os.path.join(outdir, "image_run_log.txt"))

    frames = {m: ImageFrames(img_path, m, size=args.size, gray=args.gray, bw=args.bw) for m in mods}
    f0 = frames[mods[0]]
    save_png(f0.pixels, os.path.join(outdir, "original.png"))
    with open(os.path.join(outdir, "CONDITION.txt"), "w") as fh:
        fh.write("condition: %s\nmode: %s\nimage: %s (%dx%d %s, %d bits)\nTX gains: %s dB\nRX gain: %.0f dB\n"
                 "freq: %.3f GHz, sample rate %.1f MS/s, %d samples/symbol, symbol rate %.2f Msym/s, "
                 "RRC roll-off %.2f, occupied bandwidth %.2f MHz\n"
                 % (args.tag or "-", "SIMULATION" if args.sim else "USRP B210", os.path.basename(img_path),
                    f0.shape[1], f0.shape[0], "gray" if args.gray else "RGB", f0.n_image_bits,
                    args.gains, args.rx_gain, args.freq / 1e9, f0.samp_rate / 1e6, f0.sps, f0.symbol_rate / 1e6,
                    f0.rolloff, f0.occupied_bw / 1e6))

    print("=" * 72)
    print("IMAGE TRANSFER  -  %s" % ("SIMULATION" if args.sim else "USRP B210 over the air"))
    print("image     : %s  -> %dx%d %s = %d bits" % (img_path, f0.shape[1], f0.shape[0],
                                                     "gray" if args.gray else "RGB", f0.n_image_bits))
    print("bandwidth : %.2f MHz occupied  (sample rate %.1f MS/s, %d samples/symbol, %.2f Msym/s, roll-off %.2f)"
          % (f0.occupied_bw / 1e6, f0.samp_rate / 1e6, f0.sps, f0.symbol_rate / 1e6, f0.rolloff))
    for m in mods:
        print("%s      : %d frames per image, %.3f s per image on air" %
              (m.upper(), frames[m].n_frames, frames[m].cycle_samples / frames[m].samp_rate))
    print("TX gains  : %s dB    RX gain: %.0f dB    freq %.3f GHz" % (args.gains, args.rx_gain, args.freq / 1e9))
    print("output    : %s" % outdir)
    print("=" * 72)

    sim_esn0 = args.sim_esn0_at_30 - 10 * np.log10(f0.symbol_rate / 7.5e6)   # narrower band -> less noise
    link = ImageLink(frames[mods[0]], hardware=not args.sim, freq=args.freq, tx_gain=gains[0],
                     rx_gain=args.rx_gain, sim_esn0_at_ref=sim_esn0, sim_ref_gain=30.0)
    if not args.sim:
        link.start()
        time.sleep(1.5)

    rows, images = [], {}
    captures, frame_evm = {}, {}
    try:
        for mi, m in enumerate(mods):
            tx = frames[m]
            if mi > 0:
                if not args.sim:
                    link.set_tx_on(False)
                link.set_frames(tx)
            print_header(m)
            n_cap = tx.cycle_samples + 3 * tx.frame_samples
            for g in gains:
                link.set_tx_gain(g)
                # 1) noise level: transmitter output = 0
                link.set_tx_on(False)
                noise = link.capture(300_000)
                # 2) signal: transmit the image
                link.set_tx_on(True)
                x = link.capture(n_cap)
                key = "%s_tx%02d" % (m, int(round(g)))
                captures[key] = x.astype(np.complex64)
                captures["noise_" + key] = noise[:150_000].astype(np.complex64)
                row, img, fe = analyse_point(m, g, tx, x, noise, link.last_drops, args.rx_gain, outdir)
                rows.append(row)
                images[(m, g)] = img
                if fe is not None:
                    frame_evm[(m, g)] = fe
    finally:
        if not args.sim:
            link.set_tx_on(False)
            time.sleep(0.2)
            link.stop()
            link.wait()

    np.savez(os.path.join(outdir, "captures.npz"), samp_rate=f0.samp_rate, **captures)
    write_outputs(outdir, rows, images, frame_evm, gains, mods, frames, args.gray, args.sim, args.rx_gain)


if __name__ == "__main__":
    main()
