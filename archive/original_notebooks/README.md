# Original notebooks (2025)

These are the Colab notebooks from the science fair version of ProtiQ, kept for history, with cell outputs cleared. The maintained code is the `protiq` package in `src/`.

Main changes from these notebooks:

- The training set (X, y) is now built from UniProt, where before it was never defined.
- AlphaFold features come from real models (pLDDT from the B-factor column). The notebooks used placeholder values.
- AlphaFold models are found through the AlphaFold DB API. Hardcoded `model_v4` file names now return 404 errors, because the current version is v6.
- UniProt IDs no longer have to start with P or Q.
- The Foldseek parser now reads the E-value from column 11. It used to read the bitscore from column 12 as the E-value.
- Foldseek searches structures. The old code passed it a FASTA sequence.
- Neighbor-vote features carry function information, so the kNN can learn what the similar proteins do, not just how similar they are.
