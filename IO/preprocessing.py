import numpy as np
import scipy.signal
import torch
from typing import Optional, Tuple


class BandpassResample:
    """Compile-time only (cache_dataset.py) — bandpass filter + resample to a
    fixed sample_freq. Baked into the cache once; never runs at train time."""
    def __init__(self, original_freq, sample_freq=200, l_freq=None, h_freq=None, notch_freq=None):
        self.original_freq = original_freq
        self.sample_freq = sample_freq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        if self.l_freq is not None and self.h_freq is not None:
            # A native rate at or below 2 x h_freq (e.g. 100-200 Hz vs h_freq 100) holds no content
            # above its Nyquist frequency, so there is nothing to low-pass: high-pass only. (A
            # band-pass with its upper edge clamped just under Nyquist did nothing useful and is
            # numerically fragile.) Resampling up to sample_freq cannot add content above it either.
            nyquist = self.original_freq / 2
            if self.h_freq >= nyquist:
                print(f"  [Info] native band ends at {nyquist:g} Hz (original_freq={self.original_freq}): "
                      f"high-pass {self.l_freq} Hz only, no low-pass needed.")
                sos = scipy.signal.butter(4, self.l_freq, btype='highpass', fs=self.original_freq, output='sos')
            else:
                sos = scipy.signal.butter(4, [self.l_freq, self.h_freq], btype='bandpass', fs=self.original_freq, output='sos')
            x = scipy.signal.sosfiltfilt(sos, x, axis=-1)

        if self.notch_freq is not None:
            b, a = scipy.signal.iirnotch(self.notch_freq, 30.0, fs=self.original_freq)
            x = scipy.signal.filtfilt(b, a, x, axis=-1)

        if self.sample_freq != self.original_freq:
            new_num_samples = int(x.shape[-1] * self.sample_freq / self.original_freq)
            x = scipy.signal.resample(x, new_num_samples, axis=-1)

        # sosfiltfilt/resample can hand back a negative-strided view (e.g. no
        # resample needed, filtfilt's internal reversal not copied out) — torch
        # refuses those, so force a contiguous copy before handoff.
        return torch.from_numpy(np.ascontiguousarray(x)).float()


class Normalizer:
    """Online (train-time) — normalization only. Bandpass/resample already
    baked into the compiled cache by BandpassResample, so the training
    pipeline never carries that logic. Batched: __call__ takes (N, C, T) and
    normalizes each of the N trials independently (its own mean/std,
    not one pooled across the batch) -- same per-trial semantics a Python
    loop of N single-trial calls would give, but computed in one vectorized
    pass instead. IO/dataset.py's _load_task used to do exactly that loop
    (torch.stack([transform(raw_data[i]) for i in range(N)])), which for a
    large subject (e.g. PhysionetMI's ~362 trials/subject) held N separately-
    normalized tensors in a Python list simultaneously before torch.stack
    copied them into one buffer -- a real, avoidable memory spike across a
    whole dataset's worth of subjects, not just a speed cost (see check_model.py
    OOM investigation)."""
    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        return torch.from_numpy(self._normalize(x)).float()

    @staticmethod
    def _normalize(x):
        """Per-trial z-score: x (N, C, T), one mean / std per trial over (C, T), never pooled across N."""
        mean = np.mean(x, axis=(-2, -1), keepdims=True)
        std = np.std(x, axis=(-2, -1), keepdims=True)
        return (x - mean) / (std + 1e-8)


def build_normalizer_from_config(config: dict) -> Normalizer:
    """The per-trial z-score (the only normalisation since 2026-10-07; fixed / robust / none were never used)."""
    return Normalizer()


def cache_suffix(sample_freq, bandpass_filter: dict, pre_event_seconds: float = 0.0,
                  post_event_seconds: float = 0.0, continuous_seconds: float = 0.0) -> str:
    """Derives the compiled-cache filename suffix from the params baked into it —
    shared by cache_dataset.py (writes) and IO/dataset.py (reads), so a config
    change that alters either just misses the cache instead of silently reading
    stale data. pre_event_seconds/post_event_seconds only add to the suffix when
    either is nonzero (the default 0/0 keeps existing filenames — and existing
    caches — untouched); once set, EVERY dataset's cache filename changes, same as
    a bandpass_filter change would, even datasets whose own loader ignores these
    params (they're a global compile-time setting, not a per-dataset one — see
    cache_dataset.py / IO/loader.py's cut_event_window)."""
    l_freq, h_freq = bandpass_filter['l_freq'], bandpass_filter['h_freq']
    suffix = f"fs{sample_freq}_bp{l_freq}-{h_freq}"
    if continuous_seconds:   # event-free compile: every recording cut into fixed windows, events ignored
        return suffix + f"_cont{continuous_seconds:g}"
    if pre_event_seconds or post_event_seconds:
        suffix += f"_pre{pre_event_seconds:g}_post{post_event_seconds:g}"
    return suffix


def cut_event_window(data: np.ndarray, event_pts: int, pre_pts: int, post_pts: int
                      ) -> Tuple[np.ndarray, int, int]:
    """data: [C, T] one continuous recording, event_pts: sample index of the real
    event/trigger within it -> (window [C, pre_pts+post_pts], valid_start, valid_end).

    Cuts [event_pts-pre_pts, event_pts+post_pts) around the anchor, ALWAYS returning a
    window of exactly pre_pts+post_pts samples — zero-padded on whichever side(s) run
    past the recording's actual start/end, never dropped (no threshold to clear, unlike
    the old window_continuous_signal design this generalizes — see its docstring).
    valid_start/valid_end mark the real (non-padded) content within the returned window
    (window[:, valid_start:valid_end] is real; everything outside is zero pad) — feeds
    IO/dataset.py's per-row validity, which keeps masking/loss from treating pad as
    signal.

    Padding here only covers running off the EDGE OF THE RECORDING — it does NOT
    protect against reading real content that belongs to a DIFFERENT, adjacent
    trial/event in the middle of a continuous recording (that's a labeling problem,
    not a bounds problem — see datas/finetune/PhysionetMI/loader.py's docstring for why that
    dataset doesn't use this at all)."""
    C, T = data.shape
    total = pre_pts + post_pts
    raw_start = event_pts - pre_pts
    raw_end = event_pts + post_pts
    window = np.zeros((C, total), dtype=data.dtype)
    src_start, src_end = max(0, raw_start), min(T, raw_end)
    if src_end <= src_start:
        return window, 0, 0  # entirely outside the recording -- degenerate, caller should skip
    dst_start = src_start - raw_start
    dst_end = dst_start + (src_end - src_start)
    window[:, dst_start:dst_end] = data[:, src_start:src_end]
    return window, dst_start, dst_end


def num_patches(total_T: int, patch_len: int, patch_stride: Optional[int] = None) -> int:
    """Patch count slice_patches would produce for a signal of length total_T —
    shared so callers that need P before calling slice_patches (e.g. to size a
    mask) can't drift out of sync with the real formula."""
    patch_stride = patch_stride or patch_len
    return (total_T - patch_len) // patch_stride + 1 if total_T >= patch_len else 0


def slice_patches(x: torch.Tensor, patch_len: int, patch_stride: Optional[int] = None
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """x: [..., T] (any leading dims — [C,T] per-sample or [B,C,T] batched) ->
    (x_patches [..., P, L], time_indices [P], unbatched). patch_stride=None (or
    equal to patch_len) gives the original non-overlapping behavior; smaller
    strides overlap consecutive patches within the same Window. Uses unfold
    (a view, not a copy, for the non-overlap case) so it stays generic on
    leading dims. Drops any remainder < patch_len. Shared by IO/dataset.py's
    PretrainDataset (per-sample) and the finetune RawSource
    (batched)."""
    patch_stride = patch_stride or patch_len
    T = x.shape[-1]
    if T < patch_len:
        x_patches = x[..., :0].reshape(*x.shape[:-1], 0, patch_len)
    else:
        # unfold returns a strided view (not a reshape-compatible copy) — .contiguous()
        # so downstream code (fft, model forward, collate) doesn't have to know that.
        x_patches = x.unfold(-1, patch_len, patch_stride).contiguous()  # [..., P, patch_len]
    time_indices = torch.arange(x_patches.shape[-2], dtype=torch.long)
    return x_patches, time_indices


def cut_windows(piece, win: int, min_real: float = 0.5, keep_short: bool = False):
    """Non-overlapping win-sample windows of piece [C, T] (numpy array or torch tensor) -> [(window [C, win], valid_end)];
    window[:, :valid_end] is real, the rest zero. The leftover shorter than a window is kept, zero-padded, when it holds at
    least min_real of a window (min_real 1.0: never) or, with keep_short, when it is the whole piece (a trial shorter than
    one window). The one cutter of the compile-time loaders and of window_continuous_signal."""
    n, rem = divmod(piece.shape[-1], win)
    out = [(piece[:, i * win:(i + 1) * win], win) for i in range(n)]
    if rem and (rem >= np.ceil(min_real * win) or (keep_short and n == 0)):
        zeros = torch.zeros if torch.is_tensor(piece) else np.zeros
        w = zeros((piece.shape[0], win), dtype=piece.dtype)
        w[:, :rem] = piece[:, n * win:]
        out.append((w, rem))
    return out


def window_continuous_signal(trials: torch.Tensor, target_L: int, ds_name: str, subject_id,
                              valid_ranges=None, min_real_fraction: float = 0.5,
                              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Cuts each trial's REAL content into non-overlapping target_L windows, one trial at a
    time -- never splicing two trials into one window. (It used to flatten all trials into
    one signal first, so any dataset whose trial length isn't a multiple of target_L --
    BETA 3/4 s epochs, 1 s ERP epochs, 3.1 s BCIC2020-3 trials -- got a window with the end
    of one trial glued to the start of an unrelated one, a step edge the model then
    learned to reconstruct. Measured 2026-09-24: ~100% of those datasets' windows.)

    valid_ranges: optional list of N (start, end) pairs, one per input trial, marking
    real (non-padded) content (see IO/loader.py's cut_event_window); None = every trial
    fully real. Only [start, end) of each trial is windowed, so a trial's own edge
    padding is dropped rather than carried into a window.

    A trial's leftover shorter than target_L is kept, zero-padded at the end, only if it
    holds at least min_real_fraction * target_L real samples (a whole short trial, e.g.
    a 3 s BETA epoch, counts as a leftover); otherwise it's dropped. Padding is marked
    via valid_end, so masking (and the recon loss) skip it.

    Returns (windows [n_windows, C, target_L], labels [n_windows] (dummy, always 0 --
    windows don't carry a per-trial label), valid_start [n_windows] long (always 0),
    valid_end [n_windows] long) -- window[:, :valid_end] is real, the rest is zero pad.
    """
    N, C, T = trials.shape
    if valid_ranges is None:
        valid_ranges = [(0, T)] * N
    windows, valid_ends, dropped = [], [], 0
    for i, (vs, ve) in enumerate(valid_ranges):
        real = trials[i, :, int(vs):int(ve)]
        cut = cut_windows(real, target_L, min_real_fraction)
        windows += [w for w, _ in cut]
        valid_ends += [e for _, e in cut]
        dropped += real.shape[-1] - sum(e for _, e in cut)

    if not windows:
        raise RuntimeError(f"No windows produced for {ds_name} subject {subject_id} "
                           f"({N} trials x {T}pts, target_L={target_L}, min_real_fraction={min_real_fraction}).")
    assembled = torch.stack(windows)
    n_pad = sum(v < target_L for v in valid_ends)
    print(f"  [{ds_name} S{subject_id}] {N} trials x {T}pts -> {len(assembled)} windows of {target_L}pts "
          f"({n_pad} end-padded, {dropped}pts of short leftovers dropped).")
    return (assembled, torch.zeros(len(assembled), dtype=torch.long),
            torch.zeros(len(assembled), dtype=torch.long), torch.tensor(valid_ends, dtype=torch.long))
