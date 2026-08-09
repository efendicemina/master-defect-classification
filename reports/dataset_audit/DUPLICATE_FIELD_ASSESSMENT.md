# Duplicate Field Assessment

## Observation

`ID` is the explicit issue identifier. Fields with duplicate or issue-relation semantics are:
`Blocks`, `Depends on`, `Dupe of`, `See also`. `Dupe of` is the strongest explicit
duplicate-link candidate; dependency, blocking, and see-also links may also connect related issues.
`Product` is the strongest embedded project/product-identity candidate, while `Classification`
and `Component` provide broader/finer taxonomy rather than an unambiguous catalogue project ID.
The catalogue project remains source provenance. Identifier counts are in `identifier_audit.csv`;
no rows or links were removed.

## Future methodological decision

The deduplication unit, graph grouping policy, treatment of conflicting labels, and cross-project
leakage controls must be frozen before any split is made.
