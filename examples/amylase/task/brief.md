# Project brief: amylase stain-removal predictor

| | |
| --- | --- |
| Goal | Rank alpha-amylase variants by expected stain-removal performance, so you can choose which to test in the lab. |
| Input | One amino-acid sequence per variant (425 residues, *Bacillus subtilis* alpha-amylase). |
| Output | Predicted stain-removal activity. Higher is better. This is regression. |
| Data | 3,706 measured variants: the wild type, 488 single mutants, and 3,217 variants with 2 to 8 mutations. |
| How we test it | FLIP2 "one-to-many" split as published: train on the wild type and the multi-mutants, test on the 488 single mutants. |
| How we score it | Spearman rank correlation between predicted and measured (main). NDCG, how well the top of the ranking is recovered (secondary). |
| Starting model | One-hot encoding of the sequence, ridge regression. Runs on a CPU in seconds. |
| Budget | 3 submissions. |

## Split sizes

| Set | Variants | Mutations per variant | Used for |
| --- | --- | --- | --- |
| Train | 2,574 | 0, and 2 to 8 | Fitting |
| Validation | 644 | 2 to 7 | Feedback on every submission |
| Test | 488 | 1 | One final score |

## Assumptions to confirm

- Higher activity means a better variant.
- Every sequence is unique, so no averaging of duplicates is needed.
- Validation variants are multi-mutants like the training set, while test variants are single mutants. Validation scores therefore say how well the model fits multi-mutants, which is not the same question the test asks.

## Approval

Approved by the user: worked example, written and approved by the maintainers (2026-11-11)
