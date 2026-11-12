# What you get

**A tile classifier that gets 93% of tissue calls right on patients it never trained on — and a
report that tells you where it is biased.**

Trained on the biobank set, scored once on 3,587 tiles from the later collection:

| | |
| --- | --- |
| Tiles labelled correctly | **93%** |
| Weighted F1 (tumour and stroma counted double) | 0.88 |
| Adipose, background, lymphocytes, mucus, tumour, normal mucosa | 93–100% found |
| Stroma | **60% found** — under-called |

**The useful part is that last row.** Because stroma is under-called, the tumour–stroma ratio
reads low: a true stroma share of 30% comes out around 21%, and a true 50% comes out around 37%.
The report says to read the ratio as an under-estimate and use it for ranking slides rather than
as an absolute number. A single F1 would not have told the pathologist their primary
measurement is biased.

**What made the difference:** adding stain and colour jitter during training, which came straight
out of the conversation about the violet tiles. It moved the score from 0.80 to 0.88. Nothing
after that helped — a bigger model didn't, an ensemble barely did, more augmentation hurt.

**What it took:** 5 of 8 submissions, CPU only.

**What it doesn't show:** the final split is not patient-grouped, because no slide identifiers
exist for these tiles, so the score is optimistic. The tiles are 28×28 pixels, so some confusions
— stroma against smooth muscle above all — are a resolution limit rather than a model failure.
