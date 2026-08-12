# Phase B4-H Hierarchical Multi-Task RTA AdaLoRA Feasibility

Engineering-only DEVELOPMENT-TRAIN runtime study; no validation performance was calculated. Equal one-third loss weighting is frozen to avoid a performance-tuned task-priority hyperparameter.

- Configuration: `76e17f754ad74fa5a938bb7ce180a6ed5f21cffcf7072ba1bca23111708b34fa`
- Training runtime: 6733.709 seconds
- Central three-adapter projection: 64.477 hours
- Conservative projection: 87.044 hours
- Locked test accessed: no

## Corrected Runtime Projection

The B4-H-O engineering optimization study rejected both dynamic-padding candidates for future competitive use. O1 dynamic padding plus float32 processed 0.676316942 examples/sec, and O2 dynamic padding plus bfloat16 processed 1.035119188 examples/sec, both slower than the original B4-H feasibility throughput of 4.455196891 exposures/sec over the complete three-epoch run. O2 is therefore only the best rejected optimization candidate, not the selected engineering configuration.

The corrected projection for a future full B4-H run keeps the original successful fixed-padding float32 configuration. It separates the persisted first-use epoch overhead from steady-state training by estimating steady-state epoch time as mean(epoch 2, epoch 3) = 1225.519719 seconds, startup overhead as epoch 1 - steady epoch = 3057.108725 seconds per fold, and fold training time as startup + 3 epochs at the steady-state example rate.

Corrected central total: 40.584126 hours.
Corrected conservative total with 35% margin: 54.788570 hours.
Runtime category: PRACTICAL.

The machine-readable corrected projection is persisted in `corrected_runtime_projection.json`.
