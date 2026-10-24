# TUEV -- fetched 2026-10-21, loader scaffolded (was gated, not fetched 2026-09-24)

Data on the Expansion drive (`raw` symlink), compiled: 370 subjects, 12,726 windows. Loader: one 5 s window per distinct annotated event, centred.

## Decision 2026-10-24: distinct events only (not comparable with EEG-FM-Bench's counts)

EEG-FM-Bench's 113,353 samples (87,834 / 12,473 / 13,046) equal the number of rows in all `.rec` files (checked: 518 files, 113,353
rows). A row is one (channel, event) annotation, so one event appears about 21 times; Bench makes a window per row and splits
randomly, so copies of the same window land in train and test. We keep one window per distinct (start, stop, code) event (12,726).
Our TUEV numbers are therefore not comparable with Bench's and are not inflated by duplicate windows. Class counts, ours against the
raw rows: code 6 (bckg) 4,115 / 73,372; 2: 3,173 / 15,931; 5: 2,582 / 13,257.
If a Bench-matching run is wanted later, drop the `set` in `Loader._events` (keep every row) and recompile.


Temple University EEG Events Corpus, EEG-FM-Bench member. 6-class event
classification: spike and sharp wave (SPSW), generalized periodic
epileptiform discharge (GPED), periodic lateralized epileptiform discharge
(PLED), eye movement (EYEM), artifact (ARTF), background (BCKG).

EEG-FM-Bench Table 1: 21ch, 5s windows, 87834 / 12473 / 13046
train/val/test. Split: labels are very imbalanced, so a stratified split over
ALL data at ~0.8 / 0.1 / 0.1 (not subject-disjoint).

Access: same TUH EEG form as TUAB (see ../TUAB/NOTES.md) -- one approval
covers every TUH subset. Rsync path once approved:
data/tuh_eeg/tuh_eeg_events/ (check the current version folder).

No scaffolding written yet -- revisit once access is granted.
