# Protocol V1 Preparation Report

Protocol: `eclipse-severity-v1`  
Protocol SHA-256: `85faacae1e7f411d68803653388611c37c62730ea9217e9700a0ad6ac41b7cda`

The raw sources were streamed project by project. No model was fitted and no locked-test model
performance was calculated or inspected.

## Frozen population and split

- Raw rows: 301,464
- Eligible rows: 259,473 (expected 259,473: MATCH)
- Excluded `enhancement` rows: 41,991
- Excluded missing-text rows: 0
- Development rows: 207,575
- Locked-test candidate rows: 51,898
- Candidate rows removed for exact-text overlap: 57
- Candidate rows removed for explicit duplicate-component overlap: 1,166
- Final locked-test rows: 50,675

## Integrity conclusions

- Development/locked membership overlap after purge: **0**
- Duplicate-component overlap after purge: **0**
- All per-project locked boundaries follow timestamp/issue-ID order: **PASS**
- All 27 per-project CV chronology proofs: **PASS**
- CV validation overlap removals are recorded for within-project and pooled families.
- Four known short trailing records were retained because all protocol-required fields exist.

Detailed safe aggregates are provided in the CSV and JSON files beside this report. Text and
individual locked-test membership are not committed.

```text
MODELS_FITTED = 0
TEST_SET_ACCESSED_FOR_MODEL_SELECTION = NO
LOCKED_TEST_MODEL_PERFORMANCE_ACCESSED = NO
```
