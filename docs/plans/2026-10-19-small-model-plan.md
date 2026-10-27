# Plan: the small model, the Compass suite and the ablations (2026-10-19)

**Superseded by `2026-10-21-plan.md`** (state, the TUEG corpus, order). The ablation and Compass tables below are still the reference.

The user's plan (2026-10-19): run every further ablation on the **small** model; train two more small seeds; test the
small backbones on the Compass downstream datasets; fill the method and experiment sections of the paper meanwhile; sum up
the ablations the paper needs. This file is the working list.

## 1. Backbones

| | state |
|---|---|
| `qlet_small_v4flat_nested_s1` | done (backbone_eval, 12 loso cells) |
| `qlet_small_v4flat_nested_s2`, `_s3` | running since 2026-10-19 15:41 (`output/queue/small_seeds.plan`, two at a time) |
| `qlet_small_v4flat_tr2tb10_s1-3` | matching-pursuit references (archive them once the nested seeds are in) |

GPU. One 3-seed ablation on this machine is therefore about 8-11 h wall (two slots). The 5090 server is not used (likely
CPU-bound, and the data transfer is slow). **Unit below: one small pretrain run = 5.4 h on this machine.**
(a guess is 2-3 times). **Unit below: one small pretrain run = 5.4 h on this machine.**

## 2. Compass suite on the small backbones

Decisions (user, 2026-10-19): heads trained for **10 epochs**; local datasets first; **one fixed head recipe for every
task, chosen without the reported sets**; the former DEV sets (BNCI2014009, BNCI2015001) are reported like the rest.

- Datasets: BCI cells BNCI2014001 / 004 / 008 / 009 / 2015001 and Nakanishi2015 (loso and few-shot); then EEGMat, CHB-MIT,
  SEED, Sleep-EDFx (on disk). TUAB and SEED-VIG are not on this machine (TUH registration, SEED-VIG request): added when the
  data exists. Things-EEG2 stays dropped.
- Compass splits to implement (docs/references/2601.17883_EEG-FM-Compass.md): CHB-MIT ictal + 10 min pre-ictal, few-shot =
  first seizure; Sleep-EDFx 10-fold subject split, few-shot 10 %; SEED one video per class; EEGMat few-shot 60 %; TUAB first
  3 min, 10-fold (later); SEED-VIG regression (later). MI/P300/SSVEP protocols exist (`configs/finetune_protocols.json`).
- Head recipe without the test sets: candidate recipes (the combined head with one optimiser setting versus the per-paradigm
  settings) are compared on datasets that are NOT Compass datasets: PhysionetMI, Kalunga2016, Wang2016_dev, Lee2019_ERP,
  Siena, SEED_IV. The winner is frozen before any Compass number is read. Needs a card with the candidates and the rule
  before it runs.
- 10 epochs against 50: first check on two small cells that 10 epochs matches 50 (criterion in a card).
- Cost at 10 epochs, scaled from one measured cell (BNCI2014008 loso, 33,600 trials, 33 min at 50 epochs): about 1.2 days per
  backbone for all six non-BCI sets, of which Sleep-EDFx is most.

## 3. Ablations the paper needs (small model, 3 seeds each)

Cost in small pretrain runs (SPR). "Config" = no code; "code" = a short option is needed.

| # | Ablation | Why (paper) | SPR | Needs |
|---|---|---|---|---|
| 1 | Mask mixture against random patches alone (LaBraM / CBraMod style) | method todo, Masking | 3 | config |
| 2 | Spatial terms off in training (`spatial_embedding` false) | method todo, Spatial embedding | 3 | config |
| 3 | Time position off in training | method todo, FoPE | 3 | code (FoPE is the only option now) |
| 4 | 16 and 32 qlets against 24 | appendix todo | 6 | config |
| 5 | Patch 50 against 100 (stride half) | experiments todo | 3 | config; 2 x 24 = 48 is near 50, so the qlet count may need 16 |
| 6 | Fixed Hann comb: seed 3 and the decoding cells | reviewer question | 1 | patch in `docs/cards/2026-10-18-fixed-filter-bank-option.patch`; tiny had 2 seeds |
| 7 | Dense feed-forward against the mixture | adopted without test, tiny result exists | 6 (or 3) | config (dense branch exists) |
| 8 | One training phase against two | untested by choice | 3 | config (`tokenizer_epochs` 0, check) |
| 9 | Gain-network width 16 and 32 against 8 | deferred; may explain the comb result | 6 | config |
| 10 | Sparse-montage mask against channel subsampling | paper text changed to subsampling; never evaluated | 3 | config (`montage` mode exists) |
| 11 | Stride 100 against 50 | closed unevaluated | 3 | config |
| 12 | Nested against matching pursuit on small | adoption on the small corpus | 0 more | small s2-3 running complete it |

Declined by the user: orthonormalised fixed bank, depth and width.
Priority A (the paper's own todos): 1, 2, 3, 4, 5 = 18 SPR, about 2-2.5 days here. B: 6, 9, 7, 8 = 16 SPR. C: 10, 11.
All twelve is about 38 SPR, about 4 days here.

**Evaluation only (no training), on the existing backbones, 3 seeds:** time position removed at test time (`time_off`,
`time_shuffle`), coordinates shuffled / mean / mirrored / jittered 2, 5, 10 mm, coordinate-embedding structure, the 10 and
20 s long-input check (method todos); S-star remeasure; template and decomposition figures; the source / dipole control and
the task filter rebuilt on the small nested backbones.

**Downstream ablations:** combined head against its parents (power only, signed only, latent), closed-form against SGD,
raw-EEG head against the qlet head (raw_head exists), per dataset, with the Compass suite.

## 4. Order

1. Small seeds 2-3 (running) -> Compass cells on the three small backbones as soon as the head recipe is frozen.
2. Paper: method and experiments filled while the GPU runs (no GPU needed).
3. Ablations A on small, then B, C; eval-only items any time a slot is idle.
4. Task filter, source control, figures rebuilt on the small nested backbones.
5. Rename the model; commit.
