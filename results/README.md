# Per-episode results

One JSON line per episode for the rows of the paper. Every file of a benchmark covers the same instances with the same noise
seeds, so any two rows can be compared episode by episode (`baselines/paired_stats.py`).

- **LIBERO-Plus**: the fixed 1,000 instances of `baselines/data/plus_units_selection_1000.json` (500 camera viewpoints, 500 robot
  initial states). Fields: `suite`, `task`, `name`, `base_task`, `category`, `difficulty_level`, `episode`, `seed`, `noise_seed`,
  `success`, `policy_calls`, `n_action_steps`, `steps_including_settle`, `seconds`.
- **LIBERO**: 4 suites × 10 tasks × 50 episodes. Fields: `suite`, `seed`, `task`, `episode`, `success`,
  `steps_including_settle`, `policy_calls`, `noise_seed`, `seconds`.
- **`deployment/`**: the two benchmark records behind the inference columns of Table 3 (RTX 6000 Ada, 10 denoising steps).

## GR00T N1.7, 2 bits — LIBERO-Plus (Table 1, top)

| Row | Successes | File |
|---|---:|---|
| Full precision | 618 / 1000 | `libero_plus/groot_n17/full_precision.jsonl` |
| AQLM | 401 / 1000 | `libero_plus/groot_n17/aqlm.jsonl` |
| VPTQ | 495 / 1000 | `libero_plus/groot_n17/vptq.jsonl` |
| QuIP# | 531 / 1000 | `libero_plus/groot_n17/quip_sharp.jsonl` |
| QTIP | 559 / 1000 | `libero_plus/groot_n17/qtip.jsonl` |
| Closed-loop calibration only | 575 / 1000 | `libero_plus/groot_n17/closed_loop_calibration.jsonl` |
| LoopQ | 604 / 1000 | `libero_plus/groot_n17/loopq.jsonl` |

## GR00T N1.7, 1 bit — LIBERO-Plus (Sec. 5.2)

| Row | Successes | File |
|---|---:|---|
| Closed-loop calibration only | 81 / 1000 | `libero_plus/groot_n17_1bit/closed_loop_calibration.jsonl` |
| LoopQ | 292 / 1000 | `libero_plus/groot_n17_1bit/loopq.jsonl` |

## π₀.₅, 2 bits — LIBERO-Plus (Table 1, bottom; Table 2)

| Row | Successes | File |
|---|---:|---|
| Full precision | 765 / 1000 | `libero_plus/pi05/full_precision.jsonl` |
| Closed-loop calibration only | 734 / 1000 | `libero_plus/pi05/closed_loop_calibration.jsonl` |

## π₀.₅, 1 bit — LIBERO-Plus (Table 2)

| Row | Successes | File |
|---|---:|---|
| VPTQ | 0 / 1000 | `libero_plus/pi05_1bit/vptq.jsonl` |
| QuIP# | 138 / 1000 | `libero_plus/pi05_1bit/quip_sharp.jsonl` |
| AQLM | 181 / 1000 | `libero_plus/pi05_1bit/aqlm.jsonl` |
| QTIP | 399 / 1000 | `libero_plus/pi05_1bit/qtip.jsonl` |
| Closed-loop calibration only | 450 / 1000 | `libero_plus/pi05_1bit/closed_loop_calibration.jsonl` |
| LoopQ | 498 / 1000 | `libero_plus/pi05_1bit/loopq.jsonl` |

## π₀.₅, 2 bits — LIBERO (Table 2)

| Row | Successes | File |
|---|---:|---|
| Full precision | 1928 / 2000 | `libero/pi05/full_precision.jsonl` |
| Closed-loop calibration only | 1962 / 2000 | `libero/pi05/closed_loop_calibration.jsonl` |
| LoopQ | 1951 / 2000 | `libero/pi05/loopq.jsonl` |

## π₀.₅, 1 bit — LIBERO (Table 2)

| Row | Successes | File |
|---|---:|---|
| VPTQ | 0 / 2000 | `libero/pi05_1bit/vptq.jsonl` |
| QuIP# | 496 / 2000 | `libero/pi05_1bit/quip_sharp.jsonl` |
| AQLM | 606 / 2000 | `libero/pi05_1bit/aqlm.jsonl` |
| QTIP | 1312 / 2000 | `libero/pi05_1bit/qtip.jsonl` |
| Closed-loop calibration only | 1477 / 2000 | `libero/pi05_1bit/closed_loop_calibration.jsonl` |
| LoopQ | 1600 / 2000 | `libero/pi05_1bit/loopq.jsonl` |
