"""Reading finetune runs: a run dir's group_eval.json as {subject: score} (summary, seed_equivalence and report all read runs this way), and
the cells of a head label under the groups' backbones (atom_maps, probe_maps)."""
import glob
import json
import os
import re

import numpy as np

from IO.config import QLET_OUTPUT


def load(run_dir, metric):
    p = os.path.join(run_dir, 'artifacts', 'group_eval.json')
    if not os.path.exists(p):
        return None
    subj = {}
    for fold in json.load(open(p)).values():
        for g in fold['groups'].values():
            for s, x in g['subjects'].items():
                subj.setdefault(s, []).append(x[metric])
    return {s: float(np.mean(v)) for s, v in subj.items()}


def cell_names(backbones, head, strip_seed=True):
    """Sorted cell names under output/Qlet/<backbone>/finetune/<head>/ of these backbones; strip_seed folds <cell>_seed<k> into <cell>."""
    names = {os.path.basename(d) for bb in backbones for d in glob.glob(f'{QLET_OUTPUT}/{bb}/finetune/{head}/*')}
    return sorted({re.sub(r'_seed\d+$', '', n) for n in names} if strip_seed else names)


def cell_run_dirs(bb, head, cell):
    """A backbone's run dirs of one cell: <cell> and every <cell>_seed<k> (several seeds of one cell are pooled)."""
    base = f'{QLET_OUTPUT}/{bb}/finetune/{head}/{cell}'
    return sorted(glob.glob(base) + glob.glob(base + '_seed*'))


def head_files(run_dirs):
    """Every fold's head.pth of these run dirs."""
    return [p for d in run_dirs for p in sorted(glob.glob(f'{d}/finetune/run_*/head.pth'))]
