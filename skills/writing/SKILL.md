---
name: writing
description: Write anything a The-ML-Guy user will read so that they can understand it and act on it. Use before writing a question to the user, the brief, the audit's note for the user, a hand-over or status message, the top of REPORT.md, or the text on the page.
---

# Writing for the user

The reader is a scientist. They know their problem better than you do, and they may know no
machine learning. They were not in the session: they did not see the commands, the logs or the
dead ends. They read what you write to decide something — whether to approve, whether to trust
the number, what to do next — so write for that decision.

## The rules

- **Put what they need first.** The answer, the result or the question comes in the first
  sentence. Reasons and detail follow, for the reader who wants them.
- **Use their words.** Name things the way `task/brief.md` does. When an ML term is unavoidable,
  say what it lets them do before you name it: "of the 50 it ranks highest, about 30 are really
  in your top tenth" comes before "Spearman 0.63".
- **Keep the depth, cut the padding.** Every number stays, with its unit and what it was
  measured on. Every caveat that changes what they should do stays. What goes: restating the
  question, describing your own process, a closing line that repeats the point, and words that
  only add emphasis.
- **Say one thing once.** If the report states a limit, the page and the message point to it
  rather than rephrasing it three ways.
- **Let the content choose the shape.** A table for things compared on the same attributes, a
  numbered list for steps, prose for reasoning. A short answer needs no headings.
- **Be as long as the decision needs.** A status message is a few lines. A brief is one page. If
  it grows past that, the extra belongs in a file they can open, not in what they must read.
- **Say what you do not know.** "Not tested on combinations of mutations" is more useful than
  silence, and far more useful than a confident guess.

## Before you send it

Read it once as the user would, cold:

1. Could they act on the first two sentences alone?
2. Is there a word they would have to look up, or a name you made up during the work?
3. Is there a sentence that could go without their noticing?
4. If it asks them something, is it one clear question they can answer in their own terms?

Fix what fails, then send it.
