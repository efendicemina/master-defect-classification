# Forensic Dataset Audit

This audit is descriptive. It does not define modelling eligibility, transform targets, remove
duplicates, create splits, or inspect any future locked test set.

## 1. Dataset inventory and raw row counts

**OBSERVATION:** All catalogue sources were processed independently by a strict streaming CSV
reader. Raw rows include every successfully parsed CSV record; malformed counts flag records whose
field count differs from the header. Required-field aggregates include malformed-width records
only when all required field positions remain available.

| Project | Rows | Historical | Status | Malformed | GiB |
|---|---|---|---|---|---|
| BIRT | 23308 | 23308 | MATCH | 0 | 2.56 |
| MYLYN | 13993 | 13993 | MATCH | 0 | 0.76 |
| CDT | 22371 | 22371 | MATCH | 0 | 3.05 |
| EQUINOX | 14559 | 14559 | MATCH | 1 | 1.90 |
| JDT | 63266 | 63266 | MATCH | 0 | 5.94 |
| PDE | 17639 | 17639 | MATCH | 3 | 1.24 |
| PAPYRUS | 13253 | 13253 | MATCH | 0 | 0.93 |
| PLATFORM | 122496 | 122496 | MATCH | 0 | 14.29 |
| TPTP | 10579 | 10579 | MATCH | 0 | 0.92 |

**OBSERVATION:** Four successfully parsed records have fewer fields than the 42-column header:
one in EQUINOX and three in PDE. Their record numbers, ending physical lines, and widths are in
`malformed_records.csv`. All retain the required audit-field positions and are therefore included
in target, text, timestamp, identifier, duplicate, and content aggregates; absent trailing fields
were represented as empty only for row-shape inspection.

## 2. Schema consistency

**OBSERVATION:** See `SCHEMA_AUDIT.md`, `schema_by_project.csv`, and `schema_summary.json` for the
canonical comparison. No ambiguous field was silently normalized.

## 3. Raw severity distributions

**OBSERVATION:** Values below are exact stored labels; `enhancement` is retained.

| Raw label | Count | Percentage |
|---|---|---|
| blocker | 4539 | 1.50565242 |
| critical | 9732 | 3.22824616 |
| enhancement | 41991 | 13.92902635 |
| major | 26337 | 8.73636653 |
| minor | 12059 | 4.00014595 |
| normal | 202442 | 67.15296022 |
| trivial | 4364 | 1.44760237 |
| (null/blank) | 0 | 0.0 |

**FUTURE METHODOLOGICAL DECISION:** Freeze target eligibility and S6/S3/S2 mappings only after
reviewing these raw distributions.

## 4. Missing text

**OBSERVATION:** Empty CSV fields and non-empty whitespace-only fields are counted separately.
CSV itself does not distinguish database NULL from an exported empty string.

| Project | Summary missing | Summary blank | Description missing | Description blank | Both unavailable |
|---|---|---|---|---|---|
| BIRT | 0 | 0 | 48 | 77 | 0 |
| MYLYN | 0 | 0 | 37 | 32 | 0 |
| CDT | 0 | 0 | 98 | 75 | 0 |
| EQUINOX | 0 | 0 | 127 | 33 | 0 |
| JDT | 2 | 0 | 209 | 273 | 0 |
| PDE | 0 | 0 | 198 | 81 | 0 |
| PAPYRUS | 0 | 0 | 440 | 7 | 0 |
| PLATFORM | 1 | 0 | 1761 | 751 | 0 |
| TPTP | 0 | 0 | 1 | 169 | 0 |

**FUTURE METHODOLOGICAL DECISION:** Decide text eligibility and empty-field handling before
preprocessing.

## 5. Temporal coverage

**OBSERVATION:** `Creation time` was audited as the strongest creation timestamp candidate.

| Project | Parsed | Failed | Null | Earliest | Latest |
|---|---|---|---|---|---|
| BIRT | 23308 | 0 | 0 | 2005-01-07T20:28:58+00:00 | 2022-03-24T07:45:52+00:00 |
| MYLYN | 13993 | 0 | 0 | 2004-04-15T13:51:46+00:00 | 2022-11-10T13:33:58+00:00 |
| CDT | 22371 | 0 | 0 | 2002-01-14T05:08:01+00:00 | 2022-11-25T14:42:18+00:00 |
| EQUINOX | 14559 | 0 | 0 | 2001-10-11T02:35:18+00:00 | 2022-05-20T08:05:08+00:00 |
| JDT | 63266 | 0 | 0 | 2001-10-11T02:14:41+00:00 | 2022-04-12T11:36:23+00:00 |
| PDE | 17639 | 0 | 0 | 2001-10-11T01:39:24+00:00 | 2022-11-07T11:57:16+00:00 |
| PAPYRUS | 13253 | 0 | 0 | 2008-10-06T12:39:04+00:00 | 2024-06-10T07:30:28+00:00 |
| PLATFORM | 122496 | 0 | 0 | 2001-10-11T01:34:46+00:00 | 2022-04-12T18:09:31+00:00 |
| TPTP | 10579 | 0 | 0 | 2003-08-07T21:02:12+00:00 | 2016-03-28T15:31:42+00:00 |

**FUTURE METHODOLOGICAL DECISION:** Freeze invalid-time handling and chronological boundaries
without consulting future test outcomes.

## 6. Identifier integrity and duplicate-related fields

| Project | Non-null IDs | Unique IDs | Rows in repeated IDs |
|---|---|---|---|
| BIRT | 23308 | 23308 | 0 |
| MYLYN | 13993 | 13993 | 0 |
| CDT | 22371 | 22371 | 0 |
| EQUINOX | 14559 | 14559 | 0 |
| JDT | 63266 | 63266 | 0 |
| PDE | 17639 | 17639 | 0 |
| PAPYRUS | 13253 | 13253 | 0 |
| PLATFORM | 122496 | 122496 | 0 |
| TPTP | 10579 | 10579 | 0 |

**OBSERVATION:** Explicit relation fields are documented in `DUPLICATE_FIELD_ASSESSMENT.md`.

## 7. Exact textual duplicates

**OBSERVATION:** SHA-256 hashes cover NFC-normalized raw `Summary`, a null separator, and raw
`Description`, with only null-to-empty and line-ending normalization. No semantic cleaning occurs.

| Project | Rows in exact duplicate groups | Groups | Largest |
|---|---|---|---|
| BIRT | 306 | 137 | 8 |
| MYLYN | 165 | 63 | 14 |
| CDT | 197 | 90 | 5 |
| EQUINOX | 70 | 31 | 5 |
| JDT | 428 | 197 | 7 |
| PDE | 138 | 62 | 4 |
| PAPYRUS | 54 | 25 | 3 |
| PLATFORM | 1052 | 458 | 10 |
| TPTP | 235 | 82 | 47 |

Cross-project groups and hash-only detail are in `duplicate_summary.csv` and
`duplicate_groups.csv`. Full text is deliberately not copied into reports.

**FUTURE METHODOLOGICAL DECISION:** Freeze duplicate grouping/removal and leakage policy before
splitting.

## 8. Other data quality observations

**OBSERVATION:** `text_content_signals.csv` reports fully empty rows, malformed record shapes, and
regex-based HTML, URL, email-like, and code/stack-trace indicators. These are coarse descriptive
signals, not preprocessing recommendations.

## 9. Issues requiring decisions before modelling

- Define eligible raw severity labels and then freeze S6/S3/S2 mappings.
- Define handling for empty/blank text and invalid or absent creation timestamps.
- Define exact and linked-duplicate grouping, conflicting-label handling, and split containment.
- Decide whether project identity or any metadata field is a permissible predictor.
- Freeze chronological development and locked-test boundaries in a separate reviewed task.
- Specify dataset identity/provenance hashing without placing raw or derived data in Git.

Total audit orchestration runtime: 0.113 seconds. Per-project parsing runtimes are retained
in restart checkpoints; the committed manifest records the full invocation runtime.
