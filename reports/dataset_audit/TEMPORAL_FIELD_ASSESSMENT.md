# Temporal Field Assessment

## Observation

`Creation time` is the most credible chronological field because its name denotes issue creation,
whereas `Last change time` describes later lifecycle activity and could leak post-creation
information. The audit parses `Creation time` only, using ISO-8601 semantics, and reports nulls,
failures, range, and timezone awareness per project in `temporal_coverage.csv`.

## Future methodological decision

No chronological cutoff or split has been selected. The next protocol-freezing task must decide
how timestamps, ties, invalid values, and project-specific temporal coverage affect eligibility.
