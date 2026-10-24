"""Q-atom dictionary readouts used by more than one panel: the templates D and H, and the phase-invariant similarity."""
import numpy as np


def atom_templates(model):
    """-> D, H ([n_atoms, patch_len] numpy), ids (every Q-atom)."""
    D, H = (t.detach().cpu().numpy() for t in model.atoms.templates())
    return D, H, np.arange(D.shape[0])


def atom_similarity(model):
    """Phase-invariant template similarity: a Q-atom presents a*D + b*H, so Q-atom j can show Q-atom i's
    waveform at any phase; sim(i, j) = sqrt(<D_i,D_j>^2 + <D_i,H_j>^2) in [0, 1] (1 = the same atom at
    some phase), symmetrised by max, diagonal NaN. -> (sim over the Q-atoms, their labels)."""
    D, H, ids = atom_templates(model)
    sim = np.sqrt((D[ids] @ D[ids].T) ** 2 + (D[ids] @ H[ids].T) ** 2)
    sim = np.maximum(sim, sim.T)
    np.fill_diagonal(sim, np.nan)
    return sim, [str(i) for i in ids]
