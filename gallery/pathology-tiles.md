# Tissue type from H&E tiles

## The request

> I'm a pathologist working on colorectal cancer. We want to map tissue composition across whole
> slides from new patients — the first thing we're after is the tumour-to-stroma ratio.
>
> In tiles/ I have small H&E tiles cut from slides, sorted into folders by tissue type: ADI
> adipose, BACK background, DEB debris, LYM lymphocytes, MUC mucus, MUS smooth muscle, NORM
> normal mucosa, STR cancer-associated stroma, TUM tumour epithelium. NCT-CRC-HE-100K is the big
> set from our biobank; CRC-VAL-HE-7K is a smaller set we collected later from other patients.
>
> New slides will come through the same pipeline — same lab, same scanner, same colour
> normalisation — just new patients. Around 90% correct would make me happy.
>
> Can you build that with The-ML-Guy?

## Pinning down the problem

**Do you have patient or slide identifiers for these tiles?**

> No, sorry, not any more. A random split of the 7K set is acceptable as long as you say the
> number is a bit optimistic.

This is the question that mattered. Tiles from one slide are near-duplicates, so without slide
ids the split cannot fully separate patients. The weakness went into the brief and the report
rather than being papered over.

**Which classes matter most?**

> Tumour and stroma, because the first thing we want is the tumour–stroma ratio. I'd still like
> every type roughly right. Background only needs to be filtered out.

So the metric weights tumour and stroma double — not a flat accuracy, not a flat macro average.

**Which confusions do you already expect?**

> Debris vs mucus, and debris vs necrotic tumour. Stroma vs smooth muscle can be hard at low
> resolution.

Worth asking: it tells you in advance which errors are the model's fault and which are the
problem's.

**Budget?**

> 8 attempts. About 90% would make me happy, no hard target.

### The agent came back with a picture

It rendered a sheet of sample tiles and asked about some oddly violet debris tiles it had noticed.

> Yes, those look like what our pipeline gives. The violet is the colour normalisation
> misbehaving on some necrotic areas. It happens from time to time, so new slides can have it
> too — keep them as they are.

That answer turned out to decide the project. Approved, and frozen.

## What you get

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
