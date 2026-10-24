"""Reading finetune runs: a run dir's group_eval.json as {subject: score} (summary, seed_equivalence and report all read runs this way)."""
import json
import os

import numpy as np


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
