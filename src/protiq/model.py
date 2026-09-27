"""The ProtiQ classifier, training, evaluation, ablation, and prediction."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from . import __version__
from .features import GROUPS, FeatureBuilder, group_of
from .homology import make_backend
from .structure import FoldseekBackend


class ProtiQClassifier(BaseEstimator, ClassifierMixin):
    """k-nearest neighbors on normalized, group weighted features, with a logistic
    regression fallback when the neighbors disagree (kNN confidence below a threshold).

    Defaults follow the original project: k = 7, cosine distance, distance weighting.
    ``group_weights`` sets the relative weight of sequence vs structure (and other) features.
    """

    def __init__(self, k: int = 7, metric: str = "cosine", weights: str = "distance",
                 fallback_threshold: float = 0.6, group_weights: Optional[Dict[str, float]] = None,
                 random_state: int = 42):
        self.k = k
        self.metric = metric
        self.weights = weights
        self.fallback_threshold = fallback_threshold
        self.group_weights = group_weights
        self.random_state = random_state

    def _weight_vector(self, columns: Sequence[str]) -> np.ndarray:
        gw = self.group_weights or {}
        return np.array([gw.get(group_of(c), 1.0) for c in columns], dtype=float)

    def fit(self, X: pd.DataFrame, y):
        self.columns_ = list(X.columns)
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True).fit(X.values)
        Xi = self.imputer_.transform(X.values)
        self.scaler_ = StandardScaler().fit(Xi)
        Xs = self.scaler_.transform(Xi)
        self.w_ = self._weight_vector(self.columns_)
        k = min(self.k, len(Xs))
        self.knn_ = KNeighborsClassifier(n_neighbors=k, metric=self.metric, weights=self.weights).fit(Xs * self.w_, y)
        self.lr_ = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xs, y)
        self.classes_ = self.knn_.classes_
        return self

    def _prep(self, X):
        arr = X[self.columns_].values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        return self.scaler_.transform(self.imputer_.transform(arr))

    def predict_proba_with_source(self, X):
        Xs = self._prep(X)
        p_knn = self.knn_.predict_proba(Xs * self.w_)
        p_lr = self.lr_.predict_proba(Xs)
        use_lr = p_knn.max(axis=1) < self.fallback_threshold
        proba = np.where(use_lr[:, None], p_lr, p_knn)
        return proba, np.where(use_lr, "logreg-fallback", "knn")

    def predict_proba(self, X):
        return self.predict_proba_with_source(X)[0]

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


# ------------------------------------------------------------------ config and bundle
@dataclass
class TrainConfig:
    backend: str = "auto"  # auto, diamond, kmer
    structure_search: bool = False  # Foldseek on AlphaFold models
    alphafold_features: bool = False
    physchem: bool = True
    top_n: int = 10
    k: int = 7
    metric: str = "cosine"
    fallback_threshold: float = 0.6
    group_weights: Dict[str, float] = field(default_factory=dict)
    test_size: float = 0.2
    seed: int = 42
    structure_cache: str = "data/structures"


def build_features(cfg: TrainConfig) -> FeatureBuilder:
    return FeatureBuilder(
        seq_backend=make_backend(cfg.backend),
        struct_backend=FoldseekBackend(cache_dir=cfg.structure_cache) if cfg.structure_search else None,
        use_alphafold=cfg.alphafold_features,
        use_physchem=cfg.physchem,
        top_n=cfg.top_n,
        structure_cache=cfg.structure_cache,
    )


def make_classifier(cfg: TrainConfig, **override) -> ProtiQClassifier:
    params = dict(k=cfg.k, metric=cfg.metric, fallback_threshold=cfg.fallback_threshold,
                  group_weights=cfg.group_weights or None, random_state=cfg.seed)
    params.update(override)
    return ProtiQClassifier(**params)


def top_hit_baseline(hits, labels: Dict[str, str], ids: Sequence[str], default: str) -> np.ndarray:
    """Classic annotation transfer: copy the label of the single best sequence hit."""
    return np.array([labels.get(hits[i][0].target, default) if hits.get(i) else default for i in ids])


def evaluate(y_true, y_pred, classes) -> Dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "report": classification_report(y_true, y_pred, labels=classes, zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
    }


def split_and_featurize(df: pd.DataFrame, cfg: TrainConfig, verbose: bool = True):
    """Stratified split, then features with no leakage: the reference set is the training
    split only, and training proteins never count themselves as a neighbor."""
    train, test = train_test_split(df, test_size=cfg.test_size, stratify=df["label"], random_state=cfg.seed)
    builder = build_features(cfg).fit(train)
    t0 = time.time()
    Xtr, _ = builder.transform(train["uniprot_id"], train["sequence"], exclude_self=True)
    Xte, hits_te = builder.transform(test["uniprot_id"], test["sequence"])
    Xte = Xte.reindex(columns=Xtr.columns)
    if verbose:
        print(f"Featurized {len(train)} training and {len(test)} test proteins in {time.time() - t0:.1f}s "
              f"({Xtr.shape[1]} features, backend: {builder.seq_backend.name})")
    return builder, train, test, Xtr, Xte, hits_te


def train_model(df: pd.DataFrame, cfg: TrainConfig, refit_all: bool = True, verbose: bool = True) -> Dict:
    df = df.dropna(subset=["sequence", "label"]).drop_duplicates("uniprot_id").reset_index(drop=True)
    counts = df["label"].value_counts()
    if (counts < 2).any():
        raise ValueError(f"Every class needs at least 2 proteins. Too small: {counts[counts < 2].to_dict()}")
    builder, train, test, Xtr, Xte, hits_te = split_and_featurize(df, cfg, verbose)
    clf = make_classifier(cfg).fit(Xtr, train["label"])
    proba, source = clf.predict_proba_with_source(Xte)
    y_pred = clf.classes_[proba.argmax(axis=1)]
    classes = list(clf.classes_)
    metrics = evaluate(test["label"], y_pred, classes)
    metrics["fallback_rate"] = float((source == "logreg-fallback").mean())
    majority = train["label"].value_counts().idxmax()
    metrics["baseline_majority_accuracy"] = float((test["label"] == majority).mean())
    th = top_hit_baseline(hits_te["sequence"], builder.labels, test["uniprot_id"].tolist(), majority)
    metrics["baseline_top_hit_accuracy"] = float((test["label"].values == th).mean())
    metrics["n_train"], metrics["n_test"] = len(train), len(test)

    if refit_all:  # final model uses every labeled protein as reference and training data
        builder = build_features(cfg).fit(df)
        Xall, _ = builder.transform(df["uniprot_id"], df["sequence"], exclude_self=True)
        clf = make_classifier(cfg).fit(Xall, df["label"])
        X_ref = Xall
    else:
        X_ref = Xtr
    return {
        "version": __version__,
        "config": cfg,
        "builder": builder,
        "classifier": clf,
        "columns": list(X_ref.columns),
        "X_train": clf.imputer_.transform(X_ref[clf.columns_].values),
        "classes": classes,
        "metrics": metrics,
    }


def save_bundle(bundle: Dict, path: str) -> str:
    joblib.dump(bundle, path, compress=3)
    return path


def load_bundle(path: str) -> Dict:
    return joblib.load(path)


# ------------------------------------------------------------------ experiments
def ablation(df: pd.DataFrame, cfg: TrainConfig, verbose: bool = True) -> pd.DataFrame:
    """Accuracy with each feature group removed (the ablation test from the project)."""
    builder, train, test, Xtr, Xte, _ = split_and_featurize(df, cfg, verbose)
    rows = []
    groups = [g for g in GROUPS if any(group_of(c) == g for c in Xtr.columns)]
    for removed in [None] + groups:
        cols = [c for c in Xtr.columns if removed is None or group_of(c) != removed]
        clf = make_classifier(cfg).fit(Xtr[cols], train["label"])
        pred = clf.predict(Xte[cols])
        rows.append({
            "removed_group": removed or "(none, full model)",
            "n_features": len(cols),
            "accuracy": accuracy_score(test["label"], pred),
            "macro_f1": f1_score(test["label"], pred, average="macro", zero_division=0),
        })
    out = pd.DataFrame(rows)
    out["accuracy_drop"] = out["accuracy"].iloc[0] - out["accuracy"]
    return out


def k_sweep(df: pd.DataFrame, cfg: TrainConfig, ks=(1, 3, 5, 7, 9, 11, 15, 20), verbose: bool = True) -> pd.DataFrame:
    """Accuracy as a function of k (the overfitting vs underfitting trade off)."""
    builder, train, test, Xtr, Xte, _ = split_and_featurize(df, cfg, verbose)
    rows = []
    for k in ks:
        for fb in (False, True):
            clf = make_classifier(cfg, k=k, fallback_threshold=cfg.fallback_threshold if fb else 0.0)
            pred = clf.fit(Xtr, train["label"]).predict(Xte)
            rows.append({"k": k, "logreg_fallback": fb, "accuracy": accuracy_score(test["label"], pred),
                         "macro_f1": f1_score(test["label"], pred, average="macro", zero_division=0)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ prediction
def reference_match(builder: FeatureBuilder, hits, sequence: str, min_identity: float = 99.5) -> Optional[str]:
    """Label of an (essentially) identical reference protein, if the query is one."""
    if hits and hits[0].identity >= min_identity:
        return builder.labels.get(hits[0].target)
    return None


def predict(bundle: Dict, ids: Sequence[str], sequences: Sequence[str],
            pdb_paths: Optional[Dict[str, str]] = None, n_neighbors_shown: int = 3) -> pd.DataFrame:
    """Predict function for a batch of proteins (one search call for the whole batch).

    Every row reports how much evidence backs it: a prediction with no sequence or structure
    neighbors rests on physicochemical features alone and should be treated as a weak guess.
    """
    builder: FeatureBuilder = bundle["builder"]
    clf: ProtiQClassifier = bundle["classifier"]
    ids, sequences = list(ids), list(sequences)
    t0 = time.time()
    X, hits = builder.transform(ids, sequences, pdb_paths=pdb_paths)
    X = X.reindex(columns=bundle["columns"])
    proba, source = clf.predict_proba_with_source(X)
    per_protein = (time.time() - t0) / max(len(ids), 1)
    rows = []
    for r, (pid, seq) in enumerate(zip(ids, sequences)):
        p, src = proba[r], source[r]
        seq_neigh = hits["sequence"].get(pid, [])
        struct_neigh = hits.get("structure", {}).get(pid, [])
        known = reference_match(builder, seq_neigh, seq)
        if known is not None:  # already annotated in the reference set: report that annotation
            p = np.array([1.0 if c == known else 0.0 for c in clf.classes_])
            src = "reference-match"
        best = int(p.argmax())
        if seq_neigh or struct_neigh:
            evidence = f"{len(seq_neigh)} sequence / {len(struct_neigh)} structure neighbors"
        else:
            evidence = "no homologs found: physicochemical/structure-quality features only (low evidence)"
        rows.append({
            "id": pid,
            "prediction": clf.classes_[best],
            "confidence": float(p[best]),
            "method": src,
            "evidence": evidence,
            "top_sequence_neighbors": "; ".join(
                f"{h.target} ({builder.labels.get(h.target, '?')}, {h.identity:.0f}% id)"
                for h in seq_neigh[:n_neighbors_shown]
            ),
            "seconds_per_protein": round(per_protein, 4),
            **{f"p_{c}": float(v) for c, v in zip(clf.classes_, p)},
        })
    return pd.DataFrame(rows)
