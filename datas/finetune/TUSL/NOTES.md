# TUSL -- compiled

Temple University EEG Slowing Corpus (von Weltin et al., 2017), `tuh_eeg_slowing` v2.0.1; EEG-FM-Bench member, 3 classes: seizure / slowing /
background (`cplx` would count as background; it does not occur). Loader: every recording of a patient, one 10 s window per `.tse_agg`
annotation (start, stop, label), label strings checked on the real files: `seiz`, `slow`, `bckg`.

Compiled counts: 38 patients, 299 windows (99 seizure / 100 slowing / 100 background). The raw files hold exactly 299 annotations (111 `.tse_agg`
for 112 EDFs; one EDF has no annotation file and is skipped), so the corpus is just small; EEG-FM-Bench's 210 / 43 / 37 = 290 is 9 fewer
(likely its own filtering, not checked). EEG-FM-Bench splits it stratified over all windows (not subject-disjoint), ~0.8 / 0.1 / 0.1.
