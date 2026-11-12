# What came out

One run, Claude Code. Trained on the biobank set (NCT-CRC-HE-100K), scored once on 3,587 tiles
from the later collection (CRC-VAL-HE-7K) — patients it never learned from.

**`weighted_f1 = 0.8825`**, and **93% of tiles labelled correctly** (3,341 of 3,587), against the
~90% the pathologist said would make them happy. 5 of 8 submissions used.

## The story is the augmentation

| # | What changed | weighted_f1 |
| --- | --- | --- |
| 0 | baseline | 0.6720 |
| 1 | small 3-block CNN (290k params), rotations/flips only | 0.8017 |
| 2 | **the same CNN plus stain and colour jitter** | **0.8756** |
| 3 | ensemble of 3 colour-augmented CNNs | 0.8798 |
| 4 | one wider CNN (1.1M params), colour augmentation | 0.8762 |
| 5 | 3-CNN ensemble, stronger augmentation plus zoom and blur | 0.8612 |

The whole gain is in experiment 2: **+0.074 from making the model rely less on exact colour.**
Nothing afterwards moved it — a bigger model did not, an ensemble barely did, and more
augmentation made it worse.

That change came directly out of the conversation. The agent noticed violet debris tiles, asked
about them, was told the colour normalisation misbehaves on necrotic areas *and that new slides
can have it too*, and concluded that colour was a shortcut the model must not learn. The single
most valuable modelling decision in this run is traceable to a question about an anomaly in the
data.

## What the pathologist actually got told

The report answers in the terms of the stated use — mapping tumour–stroma ratio — not in F1:

- Adipose, background, lymphocytes, mucus, tumour and normal mucosa are reliable: 93–100% found,
  93–99% of calls correct.
- **Stroma is the weak point and it is under-called:** only 60% of stroma tiles found (127 of
  210), the misses going to smooth muscle, debris and tumour.
- **Therefore the tumour–stroma ratio reads low.** A true stroma share of 30% reads as about 21%;
  a true 50% reads as about 37%. The report says to read the stroma share as an under-estimate and
  as useful for ranking slides rather than as an absolute number.

That last paragraph is the deliverable. A weighted F1 of 0.8825 would not have told the
pathologist that their primary measurement is biased downwards, or by how much.

## What not to believe

- **The final split is not patient-grouped,** because no slide or patient identifiers exist for
  these tiles. Tiles from one slide are near-duplicates, so the score is optimistic. The brief
  and the report both say so.
- **The tiles are 28×28 pixels.** Several of the confusions — stroma against smooth muscle above
  all — are partly a resolution limit, not a model failure. Full-resolution tiles are a different
  problem with a different ceiling.
- **Composition on a whole slide is not the same as accuracy on balanced tiles.** The report
  works the bias through for tumour and stroma; it does not do so for every class.
- 93% correct is on the later collection, which came from the same lab, scanner and normalisation
  pipeline. Another site is untested.
