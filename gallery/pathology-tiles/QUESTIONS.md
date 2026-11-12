# Pinning down the problem

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

## The agent came back with a picture

It rendered a sheet of sample tiles and asked about some oddly violet debris tiles it had noticed.

> Yes, those look like what our pipeline gives. The violet is the colour normalisation
> misbehaving on some necrotic areas. It happens from time to time, so new slides can have it
> too — keep them as they are.

That answer turned out to decide the project. Approved, and frozen.
