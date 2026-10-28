---
type: guide
tags: [finetune, evaluation]
---
# Finetune evaluation rules

Read before reporting or comparing finetune numbers. Audit 2026-09-24 found no label leak: subjects are disjoint across LOSO folds,
normalisation is per trial, the head's BatchNorm statistics update only in training, qlet codes come from the frozen backbone without
labels, and the score is `tail` (mean of the last 10 epochs), not a best-epoch pick on the test set.

1. **Compare runs on the same compile only.** Data compiled before 2026-09-26 was bandpassed and resampled per cut epoch, so every epoch
   carried filter edge transients (BNCI2014008: shared samples of overlapping epochs disagreed by 16% median). Numbers from before the
   recompile are not comparable with later ones.
2. **P300 windows overlap in time** (1 s windows, flashes ~0.25 s apart). A within-subject split without `purge` puts overlapping windows
   on both sides and inflates the score. `p300_fewshot` uses `purge: 5`; LOSO is unaffected.
3. **Do not use shuffled within-subject folds (`kfold`):** trials seconds apart land on both sides. Use `fewshot` (chronological
   calibration, the Compass convention) or `blocked_kfold`.
4. **One pretrain seed:** differences within about +-3 balanced-accuracy points are ties. Paired tests over subjects (Holm-corrected in
   `report`) measure subject noise, not seed noise.
5. **Keep finetune datasets out of pretraining.** BNCI2014001/004/008/009, BNCI2015001 and Nakanishi2015 never were in the corpus.
   PhysionetMI was in the v10-v13 corpus: those results are overlap-sensitive.
6. **Never report the DEV sets** (BNCI2015001, BNCI2014009): they tuned the frozen protocols.
7. **Read LOSO first, few-shot second.** Few-shot heads overfit heavily and part of a few-shot difference measures how a head copes with
   a tiny calibration set; and MI scores on Compass windows (which start at the cue) may partly read the cue response. Evidence and the
   untested check: [[2026-09-29-mi-fewshot-limits-report]].
