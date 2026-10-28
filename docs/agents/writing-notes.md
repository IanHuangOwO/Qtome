---
type: guide
tags: [guide]
---
# Writing notes

The vault is `docs/` (Obsidian). Every note is one markdown file in `docs/notes/<kind>/` (`cards`, `reports`, `decisions`, `paper`; the two index notes `reports-index` and `references-overview` sit in `docs/notes/` itself), with frontmatter and `[[wikilinks]]`. Links use the file name only, so a note can move between folders.
Tables, scripts and raw numbers a note cites go in `docs/files/<note name>/`. Figures drawn by the panels land in `docs/figures/`
(`runs/<backbone>/` for one backbone's panels, the rest at `docs/figures/<backbone-group>/`); embed them with a relative link, `![](../figures/...)`.
`docs/` is mostly git-ignored (see `.gitignore`); only `agents/` stays tracked.

## Frontmatter

```
---
type: card | report | decision | reference | guide | note
status: open | adopted | rejected | closed | finding | superseded | final | archived | review
date: YYYY-MM-DD
tags: [pretrain, masking, loss, architecture, qlets, head, finetune, noise-removal, data, evaluation, paper]
related: ["[[other-note]]"]
---
```

`status` for a card is its verdict: adopted, rejected, closed (stopped without a result), finding (a measurement, nothing to adopt), open (not run yet).
`review` marks a note whose status nobody has judged. `related` lists the notes it points at in the body (the card and its report always point at each other).

## Which kind of note

- **card** (`YYYY-MM-DD-<topic>`): one planned piece of work, written before it is done: a pre-registered experiment (question, change, criteria, then the result) or a plan for a code change (why, steps, how to check, questions). `status: open` while it is not done; there is no separate plan note, and an outdated open card is deleted.
- **report** (`YYYY-MM-DD-<topic>-report`): a comparison or pipeline run that may touch several cards; its folder is `docs/files/<same name without -report>/`.
- **decision** (`NNNN-<topic>`): rare. Only for a hard-to-reverse choice with a trade-off. Experiment verdicts stay in cards. No new one was needed since 2026-10-07.
- **reference** (`paper/`): one note per paper (`<arXiv id>_<Name>`), PDFs stay in `docs/references/`; rebuttal arguments are `paper/rebuttal-notes`.
- Rules for anyone reporting numbers are guides in `docs/agents/` (tracked), for example `finetune-evaluation`; a measured finding is a report.

Start from `docs/_templates/`, save the note in the folder of its type (Obsidian's Templates plugin reads that folder). After adding or closing a note run
`python -m tools.misc.doc_index` to rebuild `docs/Home.md`. In text, name another note as `[[note]]` or `[[note|label]]`; name a
file as a plain path. Outside `docs/` (code comments, CLAUDE.md) write the path, `docs/notes/<note>.md`.
