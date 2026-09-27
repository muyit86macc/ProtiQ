"""Synthetic protein families for offline demos and tests.

These sequences are NOT real proteins. They mimic how real data behaves (proteins come in
families of homologs that share function) so the whole pipeline can be exercised without
network access. Never report accuracy on this data as a scientific result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .sequences import STANDARD_AA

_AA = np.array(list(STANDARD_AA))


def make_synthetic_dataset(n_classes: int = 4, families_per_class: int = 6, members_per_family: int = 8,
                           length_range=(120, 400), mutation_range=(0.45, 0.85), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_classes):
        # Each class has its own mild composition bias, like real functional classes.
        bias = rng.dirichlet(np.ones(20) * 40)
        for f in range(families_per_class):
            L = int(rng.integers(*length_range))
            ancestor = rng.choice(_AA, size=L, p=bias)
            for m in range(members_per_family):
                seq = ancestor.copy()
                rate = rng.uniform(*mutation_range)
                mask = rng.random(L) < rate
                seq[mask] = rng.choice(_AA, size=int(mask.sum()), p=bias)
                s = "".join(seq)
                cut = int(rng.integers(0, max(1, L // 20)))  # small indels at the ends
                s = s[cut:] if rng.random() < 0.5 else s[: L - cut]
                rows.append({
                    "uniprot_id": f"SYN{c}F{f:02d}M{m:02d}",
                    "sequence": "M" + s,
                    "label": f"Class{c + 1}",
                    "protein_name": f"Synthetic family {c + 1}.{f + 1}",
                })
    return pd.DataFrame(rows)
