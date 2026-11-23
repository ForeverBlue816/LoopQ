<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo.svg" alt="LoopQ" width="340">
  </picture>
</p>

<h3 align="center">Quantizing in the Loop:<br>Drift-Aware Ultra-Low-Bit Vision-Language-Action Models</h3>

<p align="center"><em>Every linear layer of a VLA at 2 bits per weight, backbone and action head alike.</em></p>

---

A policy acts in closed loop: its outputs decide its next inputs. **LoopQ** quantizes a VLA with that loop in view, on top of a
trellis-coded 2-bit format:

1. **Closed-loop calibration** estimates the layer statistics on the states the quantized policy itself visits.
2. **Readout-aligned fine-tuning** keeps the 2-bit codes frozen and trains one small lookup table and one scale vector per layer,
   so that the action head reads what it reads in the full-precision model.

## Real robot

The 2-bit LoopQ π₀.₅ policy on an AgileX Piper arm.

| | Stack a cup onto a tape roll | Put a lemon into a lidded bin | Move objects into a box in clutter |
|---|:---:|:---:|:---:|
| Third-person view | [<img src="assets/demos/stack_cup_side.gif" height="130" alt="Stacking a cup onto a tape roll, third-person view">](assets/demos/stack_cup_side.mp4) | [<img src="assets/demos/lemon_bin_side.gif" height="130" alt="Putting a lemon into a lidded bin, third-person view">](assets/demos/lemon_bin_side.mp4) | [<img src="assets/demos/apple_box_side.gif" height="130" alt="Moving an apple into a box, third-person view">](assets/demos/apple_box_side.mp4) |
| Overhead camera | [<img src="assets/demos/stack_cup_top.gif" height="130" alt="Stacking a cup onto a tape roll, overhead camera">](assets/demos/stack_cup_top.mp4) | [<img src="assets/demos/lemon_bin_top.gif" height="130" alt="Putting a lemon into a lidded bin, overhead camera">](assets/demos/lemon_bin_top.mp4) | [<img src="assets/demos/apple_box_top.gif" height="130" alt="Moving a cube and an apple into a box, overhead camera">](assets/demos/apple_box_top.mp4) |

| π₀.₅ policy | Successes (of 75) | Weights on disk | GPU memory | Action chunk |
|---|:---:|:---:|:---:|:---:|
| Full precision | 59 | 6.47 GB | 6.65 GB | 81.6 ms |
| **LoopQ**, 2-bit weights, 8-bit activations | 58 | **1.58 GB** | **3.72 GB** | **74.6 ms** |

## Checkpoints

| Model | Checkpoint |
|---|---|
| π₀.₅, LIBERO, 2 bits | [ForeverBlue/LoopQ-pi05-libero-2bit](https://huggingface.co/ForeverBlue/LoopQ-pi05-libero-2bit) |
| GR00T N1.7, LIBERO, 2 bits | [ForeverBlue/LoopQ-GR00T-N1.7-libero-2bit](https://huggingface.co/ForeverBlue/LoopQ-GR00T-N1.7-libero-2bit) |

Both are calibrated and fine-tuned with LoopQ on LIBERO, and the same checkpoints run on LIBERO-Plus. They load with the stock
openpi and Isaac-GR00T loaders.

## Results

LIBERO-Plus success (%) with every linear layer at 2 bits, on 1,000 paired instances per model.

| Method | GR00T N1.7 | Camera | Init. state | π₀.₅ | Camera | Init. state |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Full precision | 61.8 | 65.2 | 58.4 | 76.5 | 74.4 | 78.6 |
| AQLM | 40.1 | 40.2 | 40.0 | 63.9 | 60.8 | 67.0 |
| VPTQ | 49.5 | 51.6 | 47.4 | 68.5 | 65.8 | 71.2 |
| QuIP# | 53.1 | 54.4 | 51.8 | 66.2 | 64.6 | 67.8 |
| QTIP | 55.9 | 54.4 | 57.4 | 72.6 | 71.0 | 74.2 |
| Closed-loop calibration only | 57.5 | 57.8 | 57.2 | 73.4 | 70.2 | 76.6 |
| **LoopQ** | **60.4** | **62.2** | **58.6** | **74.9** | **72.8** | **77.0** |

At 1 bit per weight, π₀.₅ with LoopQ completes 80.0 % of LIBERO (QTIP: 65.6 %).

## Repository

| Path | Content |
|---|---|
| [`results/`](results) | Per-episode outcomes on LIBERO-Plus and LIBERO, at 2 bits and 1 bit |
| [`baselines/`](baselines) | Evaluation protocol and baseline harness for π₀.₅ and GR00T N1.7 |
| [`deploy/tcq_kernels/`](deploy/tcq_kernels) | CUDA kernels for the 2-bit format with 8-bit activations |
| [`assets/`](assets) | Logo and real-robot clips |

```bash
cd baselines
python serve.py --checkpoint ./my_checkpoint --port 8105 &
python eval_libero_plus.py --port 8105 --units data/plus_units_selection_1000.json --out ./my_checkpoint/plus_1000.jsonl
python paired_stats.py plus --candidate ./my_checkpoint/plus_1000.jsonl --reference ../results/libero_plus/pi05/full_precision.jsonl
```
