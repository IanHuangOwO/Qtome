# TUAR -- compiled

Temple University Artifact Corpus, `tuh_eeg_artifact` v3.0.1. Not in EEG-FM-Compass or EEG-FM-Bench: used as an artifact-detection task and as a
test bed for the task filter. Loader: every recording of a patient, read whole and cut into non-overlapping 5 s windows; a window is
1 (artifact) when annotated artifact events (any channel, any type but `null`) cover at least half of it, else 0. Annotations are in the TCP
bipolar montage; the label ignores the channel and the artifact type. The windowing and label rule are my own choices (not checked against LUNA,
2026-10-24: not needed).

Compiled counts: 213 patients, 71,359 windows (43,571 clean / 27,788 artifact).
