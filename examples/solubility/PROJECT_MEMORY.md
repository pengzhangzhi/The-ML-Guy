# Project memory

Protein solubility from sequence, DeepSol split. Three submissions, CPU only, about four
minutes of training in total.

## What worked

- **Adjacent-pair (dipeptide) composition was the single biggest win:** validation accuracy
  0.6263 to 0.6878. Plain residue composition throws away all order; pair counts keep a little
  of it, at the cost of going from 21 to 421 features, which 62k training rows can carry.
- **Boosted trees over logistic regression on the same features:** 0.6878 to 0.6949. The
  features interact, and a linear model cannot use that.
- **A larger, slower-learning tree model:** 0.6949 to 0.6995. The first one was still
  improving when early stopping cut it off. The gain was small, so this is near the ceiling
  for these features.

Final: test accuracy 0.6627, balanced accuracy 0.6626, MCC 0.338, ROC AUC 0.766.

## What did not work, and why

Nothing failed outright in three submissions, but the returns fell off sharply: +0.073, then
+0.007, then +0.005. The features, not the model, are the limit. Counting residues and pairs
cannot see domain structure, disorder, or hydrophobic patches, which is what solubility
actually depends on.

## What to try with more budget

- **A pretrained protein language model.** ESM-2 embeddings with a small head is the obvious
  next step and the one most likely to matter. Published numbers for this split put ESM-1b and
  an LSTM near 0.70, so there is room, but not a great deal with a frozen encoder.
- **A class-balanced model.** Training and validation are 42% soluble while the test set is
  50/50. A model fitted to the training base rate under-calls "soluble" on a balanced set.
  Class weights, or moving the decision threshold, should gain something on test.
  **The rubric as set up will not select it:** `mlguy submit` ranks by accuracy on the
  imbalanced validation set, where a balanced model looks worse. If this project were set up
  again, the primary metric should be `balanced_accuracy` or `mcc`, which do not move with the
  base rate.
- **Longer-range features**: gapped pairs, k-mers with a reduced alphabet, or simple
  physicochemical summaries over windows.

## Things about this data worth knowing

- 62,478 / 6,942 / 2,001 sequences, 19 to 1,697 residues, standard 20-letter alphabet, no
  duplicates within a split.
- Train and validation are 42% soluble; test is deliberately balanced at 50%.
- Sequences are unrelated proteins clustered by similarity before splitting, so nothing
  position-wise applies and near-duplicate leakage between train and test is already handled.
- Labels are binary and come from expression experiments, so they carry real noise: the same
  protein can express differently between conditions. Treat small accuracy differences with
  suspicion.
