# Tissue type from H&E tiles — the request

Folders of images, nine classes. This request is unusually complete — the scientist names the
classes, says which set is which, and mentions that the second set is from other patients. What
it does not supply is the thing the agent asks for first and cannot get.

## What the scientist asked for

> Hi, I'm a pathologist working on colorectal cancer. In `tiles/` I have small H&E image tiles
> cut from slides, sorted into folders by tissue type (ADI adipose, BACK background, DEB debris,
> LYM lymphocytes, MUC mucus, MUS smooth muscle, NORM normal mucosa, STR cancer-associated
> stroma, TUM tumour epithelium). NCT-CRC-HE-100K is the big set from our biobank;
> CRC-VAL-HE-7K is a smaller set we collected later from other patients. I'd like a model that
> labels the tissue in each tile, so we can map tissue composition across whole slides from new
> patients. Can you build that with The-ML-Guy?

## What else the agent was told

- Where to work, and to build its own Python environment there.
- 4 CPUs, shared, no GPU.
- Not to modify the tool's repository, and not to look in any other trial's folder.
- To stop and ask whenever a skill said to.

## Where the data came from

PathMNIST — the 28×28 downsampling of NCT-CRC-HE-100K and CRC-VAL-HE-7K (Kather et al.
colorectal H&E tiles, nine tissue classes) — rendered out to class folders of PNGs so the agent
had to open image files rather than load a prepared array. Not redistributed here.

**The tiles are 28×28 pixels.** That is a real limit on this problem and the report is explicit
about it: at that resolution some of the distinctions the pathologist named as borderline are
genuinely hard, and the stroma/smooth-muscle confusion that dominates the error table is
partly an artefact of resolution rather than of the model.
