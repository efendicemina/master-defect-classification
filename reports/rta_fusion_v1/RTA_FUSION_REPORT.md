# Phase B3 Frozen-RTA Lexical–Semantic Fusion Report

Phase B3 reused nine persisted DEVELOPMENT-only checkpoints; no semantic embedding was regenerated and no competitive model was refitted.

## Frozen method

- Model: `Colorful/RTA` at revision `56cc614ee7cf17b8c1875e6848037f9e5bafc41a`
- Representation: `base_encoder_final_layer_first_token`, 768 dimensions, L2 normalized, semantic weight 1.0
- Base encoder frozen: yes; PEFT: no; locked test accessed: no

## Development results

| Task | Fold macro-F1 | Mean | Std | B1.5 mean | Delta | B3 fold wins |
|---|---:|---:|---:|---:|---:|---:|
| S6 | 0.2528602453|0.2768066652|0.2933756085 | 0.2743475063 | 0.0166314814 | 0.2776230020 | -0.0032754957 | 1/3 |
| S3 | 0.4570827957|0.4848489828|0.4883176949 | 0.4767498245 | 0.0139786028 | 0.4801093614 | -0.0033595369 | 0/3 |
| S2 | 0.6317775537|0.6421600370|0.6473665029 | 0.6404346979 | 0.0064800427 | 0.6421618749 | -0.0017271770 | 0/3 |

## Interpretation

- S6: B3 does not improve on B1.5 by -0.0032754957 macro-F1 (-1.180%).
- S3: B3 does not improve on B1.5 by -0.0033595369 macro-F1 (-0.700%).
- S2: B3 does not improve on B1.5 by -0.0017271770 macro-F1 (-0.269%).

B1.5 remains the development winner for S6, S3, and S2.

## Secondary comparisons

| Task | Comparator | Comparator mean | B3 mean | B3 delta |
|---|---|---:|---:|---:|
| S6 | B1.6 long-text MPNet fusion | 0.2743365317 | 0.2743475063 | +0.0000109747 |
| S3 | B1.6 long-text MPNet fusion | 0.4789472960 | 0.4767498245 | -0.0021974716 |
| S2 | B1.6 long-text MPNet fusion | 0.6401671344 | 0.6404346979 | +0.0002675635 |
| S6 | B2-Lite | 0.1870094651 | 0.2743475063 | +0.0873380413 |
| S3 | B2-Lite | 0.3924926613 | 0.4767498245 | +0.0842571631 |
| S2 | B2-Lite | 0.5700728080 | 0.6404346979 | +0.0703618899 |
| S6 | A2 classical | 0.2704255421 | 0.2743475063 | +0.0039219643 |
| S3 | A2 classical | 0.4695609418 | 0.4767498245 | +0.0071888827 |
| S2 | A2 classical | 0.6373364405 | 0.6404346979 | +0.0030982574 |
| S6 | B1 frozen MPNet | 0.2512246046 | 0.2743475063 | +0.0231229018 |
| S3 | B1 frozen MPNet | 0.4603457441 | 0.4767498245 | +0.0164040803 |
| S2 | B1 frozen MPNet | 0.5753993345 | 0.6404346979 | +0.0650353634 |

## Integrity

All nine semantic shards and all nine S6/S3/S2 × folds 1/2/3 checkpoints passed fingerprint, membership, method, checksum, frozen-model, and locked-test guards.
