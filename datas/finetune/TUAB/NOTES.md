# TUAB -- compiled

Temple University Abnormal EEG Corpus, `tuh_eeg_abnormal` v3.0.1 (NEDC rsync, raw on the Expansion drive via the git-ignored `raw` symlink).
Loader: every recording of a patient, its first 180 s, 21 sites, non-overlapping 10 s windows labelled with the recording's normal / abnormal
label (`IO/loader.py` TUH reader). Split: the corpus's own train / eval folders (see `metadata.json` runs).

Compiled counts: 2,329 patients, 2,993 recordings, 53,791 windows (27,368 normal / 26,423 abnormal). The raw corpus has exactly 2,993
recordings from 2,329 patients, so Compass's "2,383 subjects" (Table 14; 53,604 windows) is a different counting, not missing data; windows
agree within 0.35 %.

Access (done): registration form to NEDC, then ssh key + rsync; see `docs/agents/` and the memory note on the TUH fetch.
