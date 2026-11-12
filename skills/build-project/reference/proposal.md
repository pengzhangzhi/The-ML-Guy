# Research proposal

Fill in what you can and leave the rest blank; the agent will ask about anything missing.
Answer in your own words. Nothing here needs to be phrased in machine-learning terms.

## 1. What do you want to predict, and from what?

For example: "I want to predict how well an enzyme variant removes stains, from its amino-acid
sequence."

## 2. Where is the data?

Paths to files, or where they can be downloaded. Say which column is the measurement if it is
not obvious.

## 3. What will you do with what the model produces?

This decides how the model is scored, so be concrete. "I'll rank a few thousand candidates and
test the top 50 in the lab" leads to a different model from "I need the predicted value to be
within 2 units of the truth."

## 4. What will the new inputs look like, compared with the data you have?

Also decisive. For example: "I've only measured single mutants, and I want to predict
combinations of them", or "they will be variants of a different protein in the same family."
If new inputs look like the data you already have, say that.

## 5. Anything we should know about the measurements?

Noise, failed assays, duplicates, values that mean "no signal", measurements from different
batches or labs, whether a higher number is better.

## 6. How much time and compute can this use?

For example: "a couple of hours on a laptop", "one GPU overnight", "20 attempts", "done by
Friday".

## 7. Optional: what would be good enough, and do you have any preferences?

A target score, a model or paper you want followed, something that must be avoided, or an
existing approach to beat. Leave this blank if you have no view.
