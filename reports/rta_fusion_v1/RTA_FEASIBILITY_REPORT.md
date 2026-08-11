# Phase B3 — RTA Frozen Semantic Fusion Feasibility

## Scope

This phase evaluates engineering feasibility for a future development-only lexical-semantic fusion experiment using the bug-report-domain-specific `Colorful/RTA` encoder. It performed a full DEVELOPMENT tokenization audit, small MPS preflight, and deterministic 10,000-document throughput measurement. It did not materialize the full embedding cache, fit a competitive classifier, fine-tune RTA, use PEFT, or access locked-test data.

## Model provenance and representation decision

The Hugging Face repository resolved to immutable revision `56cc614ee7cf17b8c1875e6848037f9e5bafc41a`. Its model card identifies RTA as a pretrained language model for bug reports and specifies the MIT license. The pinned configuration declares `RobertaForSequenceClassification`: 12 layers, 12 attention heads, hidden dimension 768, maximum position embeddings 514, and float32 weights. The tokenizer is `RobertaTokenizer` with a native 512-token input limit.

The instantiated sequence-classification architecture has 124,647,170 parameters; its RoBERTa base encoder has 124,055,040. All parameters were frozen, leaving zero trainable parameters.

Architecture inspection found that the repository checkpoint contains an MLM head and no trained classification head. Loading the declared sequence-classification architecture therefore produces a newly initialized classifier that is not a defensible semantic representation. B3 ignores both heads and uses only the cleanly loaded pretrained RoBERTa base encoder. Because the standard RoBERTa classification head consumes the first final-layer sequence token, the single pre-specified representation is `last_hidden_state[:, 0, :]`, followed by deterministic L2 normalization. Raw and final semantic dimensions are both 768. No pooling comparison was conducted.

## Frozen future experiment

The future matrix, if separately authorized, remains exactly S6/S3/S2 × three persisted frozen folds. It uses the B1.5 lexical selections: S6 CHAR + LinearSVC balanced C=0.25, S3 WORD + LogisticRegression balanced C=2.0, and S2 WORD_CHAR + LogisticRegression balanced C=2.0. Semantic weight is fixed at 1.0. Canonical Summary + Description is tokenized using RTA's standard 512-token truncation with no new cleaning, metadata, ablation, or chunking.

The implementation reserves deterministic project-shard and task/fold checkpoint paths with fail-closed provenance validation. `--stage full` remains authorization-gated and fails before I/O in this feasibility phase.

## DEVELOPMENT tokenization audit

| Quantity | Result |
| --- | ---: |
| Total records | 207,575 |
| Native maximum | 512 tokens |
| Fit without truncation | 176,022 (84.7992%) |
| Require truncation | 31,553 (15.2008%) |
| Mean tokens | 439.319 |
| Median tokens | 145 |
| P90 | 920 |
| P95 | 1,751 |
| P99 | 4,892.52 |
| Maximum | 139,824 |

The highly right-skewed length distribution means the median report is short while a small tail is extremely long. B3 intentionally uses native truncation rather than introducing another long-text method. Project-level aggregate counts are preserved in `tokenization_audit.csv`; no text or token IDs are reported.

## MPS preflight

The exact pinned tokenizer and model loaded successfully. The RoBERTa base encoder ran in evaluation and inference modes on Apple MPS with float32. Repeated identical inputs produced deterministic vectors within the fixed numerical tolerance; batch inference, representative truncation, finiteness, 768-dimensional output, freezing, and L2 normalization passed.

The initial conservative batch size was 16. One engineering-only batch-32 probe succeeded, so batch size 32 was selected. There were no OOM events, batch reductions, unsupported MPS operations, or CPU fallbacks.

## Throughput measurement

A deterministic stable-identity hash selected exactly 10,000 DEVELOPMENT documents. Tokenization took 2.412 seconds, RTA encoding took 390.928 seconds, and L2 normalization took 0.015 seconds. Total measured runtime was 393.355 seconds.

End-to-end throughput was 25.4223 documents/second; encoder-only throughput was 25.5801 examples/second. Peak process RSS was approximately 1.90 GB, and MPS driver allocation after measurement was approximately 2.17 GB. These are engineering observations rather than guaranteed unified-memory peaks.

## Runtime and storage projection

| Component | Seconds | Hours |
| --- | ---: | ---: |
| Full RTA DEVELOPMENT embedding generation | 8,165.070 | 2.2681 |
| Nine fusion fit/prediction operations | 1,731.289 | 0.4809 |
| TF-IDF, validation, orchestration, reporting | 652.532 | 0.1813 |
| **Central total** | **10,548.891** | **2.9302** |
| **Conservative upper bound** | **13,866.536** | **3.8518** |

The central embedding projection applies measured end-to-end throughput to all 207,575 records. Fusion and overhead reuse empirical B1.5 timing because RTA has the same 768-dimensional normalized semantic block. The upper bound assumes 75% of measured throughput and a 25% fusion/overhead margin; neither estimate is exact.

The raw `207,575 × 768 × float32` matrix requires 637,670,400 bytes (608.13 MiB). A project-sharded normalized Parquet cache is expected to be approximately 591,816,188 bytes (564.40 MiB), with an approximate metadata-inclusive upper bound of 642,670,400 bytes (612.90 MiB). Sequential project materialization should be technically safe on the M5/16-GB machine given the observed memory behavior, provided the full run retains only one project/batch working set at a time.

```text
FULL_B3_RUN_STARTED = NO
FULL_RTA_EMBEDDINGS_MATERIALIZED = NO
COMPETITIVE_FITS_STARTED = 0
RTA_FINE_TUNED = NO
PEFT_USED = NO
MODEL_SELECTION_DATA = DEVELOPMENT_ONLY
LOCKED_TEST_TOKENIZED = NO
LOCKED_TEST_EMBEDDED = NO
LOCKED_TEST_MODEL_PERFORMANCE_ACCESSED = NO
LOCKED_TEST_USED_FOR_TUNING = NO
```
