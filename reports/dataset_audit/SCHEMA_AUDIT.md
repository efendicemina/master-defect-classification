# Schema Audit

## Observation

The audit found 42 columns present in every project. Delimiter and encoding details,
column order, duplicate names, missing columns, and project-unique columns are recorded in
`schema_summary.json`. Field names were preserved exactly; no semantic normalization was applied.

Likely semantic roles were assessed explicitly: `Severity` (raw target), `Summary` and
`Description` (candidate text), `Creation time` (candidate chronology), and `ID` (issue identity).

## Future methodological decision

The modelling schema and permitted predictors must be frozen separately. In particular, the
presence of a field does not authorize its use as a feature.
