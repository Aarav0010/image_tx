# Image transmission over BPSK / QPSK at 5.2 GHz (USRP B210 + GNU Radio)

B.Tech Project: Aarav (2301EE01), guide Dr. Jagannath, Dept. of Electrical Engineering, IIT Patna.

Sends a still image over the air with one **USRP B210** at **5.2 GHz**, using **BPSK** and **QPSK** in a
**10 MHz** channel, and rebuilds the image at the receiver. For each USRP TX gain (default 20, 22, ... 30 dB)
it measures the noise level, received power, SNR, bit error rate (BER) and image quality (PSNR), and saves
the received picture. It also runs in simulation without any hardware.

## Files

| File | What it does |
|---|---|
| `image_link.py` | Builds the frames from the image, the GNU Radio TX/RX flowgraph, and the receiver |
| `run_image_transfer.py` | Main script: sweeps the TX gains (hardware or `--sim`) and saves the results |
| `reanalyse_image_run.py` | Decodes a saved run again from its raw samples (`captures.npz`), no hardware |
| `psk_common.py` | Shared helpers: RRC filter, preamble, BPSK/QPSK mapping, theory BER |
| `image.jpg` | Test image (any image works with `--image`) |
| `10_RUN_IMAGE_TRANSFER_HW.bat` | Windows: run on the B210 (asks condition name, RX gain, TX gains) |
| `11_RUN_IMAGE_TRANSFER_SIM.bat` | Windows: run in simulation |
| `12_REANALYSE_IMAGE_RUN.bat` | Windows: re-decode a saved run |

## Setup on a new computer

1. **Install GNU Radio with UHD.** Easiest on Windows: [radioconda](https://github.com/ryanvolz/radioconda)
   (includes GNU Radio 3.10, UHD, numpy, matplotlib, pillow). On Linux: `sudo apt install gnuradio uhd-host python3-pil python3-matplotlib`.
2. **Download the USRP firmware images** once: `uhd_images_downloader`.
3. **Connect the B210**: power adapter, then USB 3.0. Check it is found with `uhd_find_devices`
   (run it twice if the first time says "No UHD Devices Found" while it loads firmware).
4. **Antennas (5 GHz)**: TX antenna on **RF0 "TX1 RX1"**, RX antenna on **RF1 "RX2"**.
5. Close uhd_fft / GNU Radio Companion before running (only one program can use the B210).

The `.bat` files look for radioconda's `python.exe` in `%USERPROFILE%\radioconda`, `C:\radioconda` or
`D:\BTP\radioconda`. If yours is elsewhere, set `RADIOCONDA_PY` to its `python.exe`, or run the commands
below from the "Radioconda Prompt".

## How to run

```
python run_image_transfer.py --sim                       # simulation, no hardware (about 1 minute)
python run_image_transfer.py --tag clear                 # B210: TX gain 20..30 dB, RX gain 40 dB
python run_image_transfer.py --tag barrier --gains 30,35,40,45,50
python run_image_transfer.py --mod bpsk --gray --image my_photo.png --rx-gain 50
python reanalyse_image_run.py results_image_<date>_<time>_10MHz_clear
```

| Option | Meaning |
|---|---|
| `--sim` | Simulated channel instead of the B210 |
| `--tag NAME` | Condition name, added to the results folder name |
| `--gains 20,22,...` | USRP TX gains in dB |
| `--rx-gain 40` | USRP RX gain in dB |
| `--mod both / bpsk / qpsk` | Modulations to test |
| `--bw 10 / 3.4` | Occupied bandwidth in MHz (10 = default) |
| `--image FILE`, `--size 128`, `--gray` | Image to send, resized to size × size, colour or grayscale |
| `--freq 5.2e9` | Carrier frequency in Hz |

Each run writes a folder `results_image_<date>_<time>_<bw>_<tag>/` with the received images
(`rx_<mod>_txXX.png`), `fig_img1_received_images.png` (all images side by side),
`fig_img2_noise_snr_vs_txgain.png`, `fig_img3_ber_psnr.png`, `fig_img4_per_frame_quality.png`,
`image_results.csv`, the log, and `captures.npz` (raw received samples, for `reanalyse_image_run.py`).

## How it works

**Link settings**

| Parameter | Value |
|---|---|
| Carrier | 5.2 GHz |
| Sample rate | 15 MS/s, 2 samples/symbol |
| Symbol rate | 7.5 Msym/s |
| Pulse shape | root-raised cosine, roll-off 0.33 |
| Occupied bandwidth | 7.5 × 1.33 = 9.97 MHz |
| Bit rate | BPSK 7.5 Mb/s, QPSK 15 Mb/s |

(`--bw 3.4`: 10 MS/s, 4 samples/symbol, 2.5 Msym/s, roll-off 0.35, 3.4 MHz.)

**Image to frames.** The image's raw pixel values (8 bits per colour value, not the JPEG file) are turned
into bits, so a wrong bit spoils only one pixel and noise shows up as visible dots. The bits are cut into
frames:

```
[ 16 guard | 127-symbol BPSK preamble | 64-symbol header (frame number) | 1024 data symbols (BPSK or QPSK) ]
```
A 128×128 colour image = 393,216 bits = 384 BPSK frames (63 ms) or 192 QPSK frames (32 ms). The image is
sent again and again in a loop.

**GNU Radio flowgraph** (built in `image_link.py`):
```
TX:  Vector Source (image frames, repeating) -> RRC Interpolating FIR (x2) -> Multiply Const -> USRP Sink (TX1 RX1)
RX:  USRP Source (RX2) -> Copy (on only while capturing) -> Vector Sink
SIM: ... Multiply Const -> Channel Model (noise, frequency offset) -> Vector Sink
```

**Measurement at each TX gain.** (1) TX output set to zero, record the noise level; (2) TX on, record one
full image; (3) decode and save the image; (4) BER against the original bits, SNR from the error vector
(EVM), PSNR against the original image.

**Receiver** (Python/numpy): matched filter, preamble search over all frames, re-sync on every frame (handles
USB sample drops), fine fractional timing, frequency offset (robust to interference bursts), per-frame phase,
header decode (frame number), bit decisions, image rebuild. Theory for both BPSK and QPSK:
BER = Q(√(2·Eb/N0)).

**Note on 5.2 GHz:** it is also WiFi channel 40. Nearby WiFi can hit a few milliseconds of a capture and
show up as horizontal stripes in the image; the log reports it in the `intf` and `bad` columns.
