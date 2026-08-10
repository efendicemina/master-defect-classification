# Phase B1.6 — Long-Text-Aware MPNet Fusion Feasibility

## Scope

This phase measured development-only workload and runtime feasibility. It did not materialize the full long-text cache, fit a classifier, execute any of the nine future competitive fits, or access locked-test text or model performance. Protocol v1 and the authenticated A2, B1, and B1.5 reports remained unchanged.

The frozen encoder is `sentence-transformers/all-mpnet-base-v2` at revision `e8c3b32edf5434bc2275fc9bab85f82640a19130`, in evaluation mode with gradients disabled. The measurement used Apple MPS and float32.

## Fixed long-text representation

Canonical Summary + Description is tokenized without truncation. Each full MPNet input contains 382 content tokens plus MPNet's two special tokens, for exactly 384 model tokens. Consecutive content windows overlap by 64 tokens and advance by 318 tokens. The last shorter chunk is retained, and no maximum chunk count is imposed.

Each chunk receives a frozen, L2-normalized 768-dimensional MPNet embedding. Document vectors are the arithmetic mean of their chunk vectors followed by final L2 normalization. Pooling, overlap, chunk length, semantic weight 1.0, and downstream B1.5 configurations are fixed rather than tuned.

## Full development chunk audit

| Chunk count | Documents |
| --- | ---: |
| 1 | 164,423 |
| 2 | 16,837 |
| 3 | 6,392 |
| 4 | 4,452 |
| 5 | 3,464 |
| >5 | 12,007 |
| **Total** | **207,575** |

Mean chunks per document were 1.962804, the median was 1, and the maximum was 479. The full representation would encode 407,429 chunks: 199,854 more than B1's one-sequence-per-document workload, an increase of 96.28%. The 43,152 multi-chunk documents equal the B1 MPNet truncation count, providing a consistency check. Project-level counts are in `chunking_audit.csv`; no raw text or token IDs are persisted.

## MPS throughput measurement

Deterministic stable-identity hashing selected 5,004 documents containing 10,011 chunks. Its mean of 2.0006 chunks/document is close to the full workload mean of 1.9628. Tokenization and chunk construction took 1.313 seconds, MPNet encoding took 431.534 seconds, and document pooling took 0.018 seconds. Total benchmark time was 432.866 seconds.

Measured encoding throughput was 23.1986 chunks/second; end-to-end document throughput was 11.5602 documents/second. Batch size remained 64. There were no MPS fallbacks, OOM events, or batch reductions. Peak process RSS was approximately 1.05 GB and post-measurement MPS driver allocation was approximately 3.24 GB.

## Runtime projection

At measured chunk throughput, full semantic materialization is projected at 17,562.640 seconds (4.8785 hours). The empirical B1.5 fit/prediction component contributes 1,731.289 seconds (0.4809 hours), while its remaining TF-IDF construction, validation, orchestration, and reporting overhead contributes 652.532 seconds (0.1813 hours). The central total is therefore 19,946.461 seconds (5.5407 hours).

The reasonable upper bound is 26,396.630 seconds (7.3324 hours), based on 75% of measured chunk throughput and a 25% margin on fusion and overhead. This is an engineering projection, not a guarantee.

## Disk and memory

The uncompressed pooled float32 matrix is 637,670,400 bytes (608.13 MiB). Because the row count, dimension, dtype, normalization, and project-sharded Parquet format match B1 E2, the central cache estimate is the measured B1 E2 size, 591,816,188 bytes (564.40 MiB), with an approximate upper bound of 642,670,400 bytes (612.90 MiB) including metadata allowance. The full cache is materially similar to the existing B1 MPNet cache, not larger by the chunk multiplier, because only one pooled vector per document is persisted.

Observed memory use and the project-sharded sequential design indicate that an M5-class system with 16 GB unified memory is technically safe. The future materializer must encode and discard chunk batches per project, persist one authenticated shard at a time, and never retain all chunk embeddings globally. The 479-chunk maximum is reported rather than capped.

## Future authorization-gated run

The CLI reserves `defect-classifier run-long-text-fusion --stage full --resume`, Git-ignored project-shard cache paths, and deterministic task/fold checkpoint paths. In this feasibility build, `--stage full` fails closed before any I/O because the competitive experiment is not authorized. A later explicitly authorized change can connect these prepared cache/checkpoint interfaces to the existing B1.5 sparse fusion machinery and detached launchd/caffeinate wrapper without changing the frozen method.

The future matrix remains exactly S6/S3/S2 × folds 1–3. It reuses S6 CHAR + LinearSVC balanced C=0.25, S3 WORD + LogisticRegression balanced C=2.0, S2 WORD_CHAR + LogisticRegression balanced C=2.0, and semantic weight 1.0.

```text
FULL_B1_6_RUN_STARTED = NO
FULL_LONG_TEXT_EMBEDDINGS_MATERIALIZED = NO
COMPETITIVE_FITS_STARTED = 0
TRANSFORMER_ENCODER_FINE_TUNED = NO
MODEL_SELECTION_DATA = DEVELOPMENT_ONLY
LOCKED_TEST_TOKENIZED = NO
LOCKED_TEST_EMBEDDED = NO
LOCKED_TEST_MODEL_PERFORMANCE_ACCESSED = NO
```
