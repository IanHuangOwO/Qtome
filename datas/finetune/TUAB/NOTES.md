# TUAB -- fetched 2026-10-21, loader scaffolded (was gated, not fetched 2026-09-22)

Access granted; data on the 5090 (F:\tuh_data\tuh_eeg_abnormal\v3.0.1), not yet copied here. gen_metadata.py / loader.py written from
the corpus documentation and tested on TUEG files (IO/tuh.py); run gen_metadata.py once raw/edf is in place and check counts
(Compass: 2,383 subjects, 53,604 windows).


Temple University Abnormal EEG Corpus. Table 14: 2383 subjects, 21ch, 250Hz,
10s trials, 53604 trials, normal/abnormal classification.

Access: registration form (isip.piconepress.com/projects/nedc/forms/
tuh_eeg.pdf), email signed copy to help@nedcdata.org, subject "Download The
TUH EEG Corpus". Approval ~24-48h, then ssh key + rsync (no plain
username/password). Rsync path once approved:
data/tuh_eeg/tuh_eeg_abnormal/v3.0.1

No public mirror found. User needs to submit the form; scaffolding
(gen_metadata.py/loader.py) can be written once access is granted or
against the corpus's documented format if useful sooner.
