# TUEV -- fetched 2026-10-21, loader scaffolded (was gated, not fetched 2026-09-24)

Data on the 5090 (F:\tuh_data\tuh_eeg_events\v2.0.1), not yet copied here. Loader: one 5 s window per distinct annotated event, centred;
check against EEG-FM-Bench / BIOT before reporting numbers.


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
