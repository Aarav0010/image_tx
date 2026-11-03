#!/usr/bin/env python3
"""
reanalyse_image_run.py - decode again the raw captures saved by run_image_transfer.py
(captures.npz) with the current receiver. No hardware needed.

    python reanalyse_image_run.py results_image_20261102_0149_10MHz_clear

Results go to <run folder>\\reanalysed\\ (same files as a normal run).
If the run has no noise captures (runs before 2 Nov 2026), the noise level is estimated from the part
of the spectrum outside our 10 MHz signal (noise is flat over the 15 MHz sampled band).
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_image_transfer as rit
from image_link import ImageFrames, WIN


def noise_from_oob(x, samp_rate, signal_bw):
    """Noise power (full band) from the FFT bins between the signal edge and the sampling edge."""
    nw = len(x) // WIN
    X = np.fft.fftshift(np.fft.fft(x[: nw * WIN].reshape(nw, WIN), axis=1), axes=1)
    f = np.fft.fftshift(np.fft.fftfreq(WIN, 1.0 / samp_rate))
    oob = (np.abs(f) > 0.56 * signal_bw) & (np.abs(f) < 0.45 * samp_rate)
    return float(np.median(np.mean(np.abs(X[:, oob]) ** 2, axis=1)) / WIN)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    run = os.path.abspath(sys.argv[1])
    cond = open(os.path.join(run, "CONDITION.txt")).read()
    bw = "10" if "15.0 MS/s" in cond else "3.4"
    gray = "gray" in cond.split("image:")[1].split("\n")[0]
    rx_gain = float(re.search(r"RX gain: ([\d.]+)", cond).group(1))
    sim = "SIMULATION" in cond
    img_path = os.path.join(HERE, "image.jpg")
    outdir = os.path.join(run, "reanalysed")
    os.makedirs(outdir, exist_ok=True)
    sys.stdout = rit.Tee(os.path.join(outdir, "image_run_log.txt"))
    print("RE-ANALYSIS of %s with the current receiver" % os.path.basename(run))
    print(cond.strip())

    d = np.load(os.path.join(run, "captures.npz"))
    keys = [k for k in d.files if re.match(r"^(bpsk|qpsk)_tx\d+$", k)]
    mods = [m for m in ("bpsk", "qpsk") if any(k.startswith(m) for k in keys)]
    gains = sorted({float(k.split("_tx")[1]) for k in keys})
    frames = {m: ImageFrames(img_path, m, gray=gray, bw=bw) for m in mods}
    rit.save_png(frames[mods[0]].pixels, os.path.join(outdir, "original.png"))

    rows, images, frame_evm = [], {}, {}
    for m in mods:
        tx = frames[m]
        rit.print_header(m)
        for g in gains:
            key = "%s_tx%02d" % (m, int(round(g)))
            if key not in d.files:
                continue
            x = d[key].astype(complex)
            if "noise_" + key in d.files:
                noise = d["noise_" + key].astype(complex)
            else:   # older run: synthesize a noise record with the out-of-band noise level
                p = noise_from_oob(x, tx.samp_rate, tx.occupied_bw)
                rng = np.random.default_rng(0)
                noise = np.sqrt(p / 2) * (rng.standard_normal(150_000) + 1j * rng.standard_normal(150_000))
            row, img, fe = rit.analyse_point(m, g, tx, x, noise, 0, rx_gain, outdir)
            row["usb_drops"] = ""
            rows.append(row)
            images[(m, g)] = img
            if fe is not None:
                frame_evm[(m, g)] = fe
    rit.write_outputs(outdir, rows, images, frame_evm, gains, mods, frames, gray, sim, rx_gain)


if __name__ == "__main__":
    main()
