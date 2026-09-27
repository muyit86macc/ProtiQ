# Benchmarks, 27 September 2026

All numbers are from `protiq` v0.2.0 on data downloaded from UniProt and the AlphaFold DB on 27 September 2026. The split is a stratified random 80/20 with seed 42, and all runs use `--backend diamond` (DIAMOND 2.1.9, `--sensitive`) unless noted. Rerun these with the commands shown. Because UniProt changes over time, your numbers may drift slightly.

## Datasets

| Dataset | Source query (all reviewed, non-fragment, length 50 to 1500) | Proteins |
|---|---|---|
| Enzyme class | Bacteria (taxon 2), up to 260 per EC class plus non-enzymes. Multi-class EC entries dropped. | 1,847 (8 classes) |
| Virulence | All bacterial proteins with keyword KW-0843 (Virulence), plus non-virulence proteins drawn evenly from 15 pathogen taxa | 5,142 (2,065 / 3,077) |
| PA14 targets | *P. aeruginosa* UCBPP-PA14 (taxon 208963), protein name "uncharacterized protein" | 326 |

## Main results

| Task | Model | Accuracy | Macro F1 | Top BLAST hit | Majority |
|---|---|---|---|---|---|
| Virulence | ProtiQ (DIAMOND) | **90.0%** | 0.894 | 87.8% | 59.9% |
| Enzyme class | ProtiQ (DIAMOND + AlphaFold features) | **68.9%** | 0.683 | 65.1% | 14.1% |
| Enzyme class | ProtiQ (DIAMOND) | 67.8% | 0.674 | 65.1% | 14.1% |
| Enzyme class | ProtiQ (k-mer backend, no install) | 51.1% | 0.499 | 39.5% | 14.1% |

"Top BLAST hit" copies the label of the single best sequence hit, which is the classic manual approach. "Majority" always predicts the most common class.

Across three seeds on the enzyme-class task, ProtiQ averaged 66.1% with AlphaFold features and 66.4% without. The best-hit baseline averaged 64.1%. The difference from AlphaFold summary features is within noise.

## k-sweep (virulence)

```
k    kNN only   kNN + LR fallback
1    0.870      0.870
3    0.893      0.895
5    0.893      0.896
7    0.900      0.900
9    0.898      0.903
11   0.902      0.901
15   0.900      0.904
20   0.896      0.902
```

k = 1 overfits. Accuracy plateaus from k = 7, consistent with the original project's k = 5 to 7 optimum.

## Ablation

Enzyme class, with DIAMOND and AlphaFold features:

```
removed group      accuracy   drop
(none)             0.689
sequence           0.435      0.254
alphafold          0.678      0.011
physchem           0.657      0.032
```

Virulence, with DIAMOND:

```
removed group      accuracy   drop
(none)             0.900
sequence           0.863      0.037
physchem           0.872      0.028
```

Sequence neighbors are the strongest signal. AlphaFold summary features (pLDDT confidence and compactness) contribute little. Testing the original project's structural claim properly requires Foldseek structural-neighbor features. That code is implemented and tested, but a benchmark needs AlphaFold models for every reference protein. It's the next experiment.

## PA14 batch prediction

- 326 proteins in 2.1 seconds, about 0.006 seconds per protein, with the batch DIAMOND search.
- Only 13 of 326 had a sequence homolog in the 5,142-protein virulence reference. The other 313 predictions rest on physicochemical features alone and are marked as low evidence in the output.
- Example: A0A0H2ZCC1 is 99.5% identical to Tse4 (Q9I069), a type VI secretion toxin of *P. aeruginosa* PAO1, and is predicted Virulence.

## Caveats

- A random split lets close homologs appear in both train and test, which inflates accuracy. A homology-clustered split is on the roadmap.
- UniProt's first results for a query are dominated by model organisms (E. coli K-12, B. subtilis). The virulence set counters this with organism-matched negatives. The enzyme-class set doesn't yet.
- The virulence keyword is incomplete, so some "non-virulence" proteins are probably unannotated virulence factors.
