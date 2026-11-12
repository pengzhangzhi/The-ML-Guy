# What the agent asked, and what it was told

The answers are quoted as the scientist sent them. The questions are condensed from longer
versions that each carried a proposed answer.

## The five questions

**1. Do you have patient or slide identifiers for these tiles?**

> No, I don't have a patient or slide list for these tiles any more, sorry. A random split of the
> 7K set is acceptable as long as you say the number is a bit optimistic.

This is the one that matters, and the answer is **no**. Tiles from one slide are near-duplicates
of each other, so without slide ids there is no way to guarantee the final split separates
patients. The scientist accepted a known-weaker split *with the weakness written into the
report* — which is the right trade and is only available if the question gets asked. Both the
brief and the report say the number is optimistic.

**2. Will new slides come through the same pipeline?**

> Same pipeline: same lab, same scanner, same tile size and colour normalisation. New patients.

**3. Which classes matter most?**

> Tumour and stroma matter most to me, because the first thing we want from the maps is the
> tumour–stroma ratio. But I'd still like every type to be roughly right. Background only needs
> to be filtered out.

This produced a weighted metric with tumour and stroma counted double — not a plain accuracy and
not a flat macro average, but the shape the stated use implies.

**4. Budget, and what would be good enough?**

> 8 attempts is fine. About 90% would make me happy; no hard target.

**5. Which confusions are expected?**

> Debris vs mucus and debris vs necrotic tumour are the usual borderline calls. Stroma vs smooth
> muscle can also be hard at low resolution.

The scientist predicted the exact failure the final model had. Stroma/smooth-muscle is the
dominant error in the confusion table. Worth asking for, because it tells you in advance which
errors are the model's fault and which are the problem's.

## The second exchange: the agent showed the scientist their own data

The agent rendered a contact sheet of sample tiles and asked whether they looked like the
pathologist's own pipeline output — including some oddly violet debris tiles it had noticed.

> I looked at the sheet. Yes, the right-hand tiles look like what our pipeline gives. The violet
> debris is almost certainly the colour normalisation misbehaving on some necrotic areas. It does
> happen in our pipeline from time to time, so new slides can have it too, and I agree with
> keeping them as they are.
>
> The weighting with tumour and stroma counted double is exactly what I want. Approved, please go
> ahead.

The violet tiles turned out to be the whole story of the project: see [RESULT.md](RESULT.md).

## What to take from this

- **Say whether you have grouping identifiers (patient, slide, donor, batch) before anything
  else.** If you do not, say so — the split can be built honestly around the gap, but only if
  the agent knows.
- **Predict your own hard cases.** It costs a sentence and it separates model failure from
  problem difficulty when the report arrives.
- **Look at the pictures the agent sends you.** An anomaly the agent cannot interpret (violet
  tiles) may be something you recognise instantly, and here it decided what the model had to be
  robust to.
