"""LIME explanations: which features, and which neighboring proteins, drove a prediction."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:  # pragma: no cover
    LimeTabularExplainer = None


def explain_prediction(bundle: Dict, protein_id: str, sequence: str, num_features: int = 10,
                       num_samples: int = 2000, html_path: Optional[str] = None,
                       pdb_paths: Optional[Dict[str, str]] = None) -> Dict:
    if LimeTabularExplainer is None:
        raise ImportError("LIME is not installed. Run: pip install lime")
    builder, clf = bundle["builder"], bundle["classifier"]
    X, hits = builder.transform([protein_id], [sequence], pdb_paths=pdb_paths)
    X = X.reindex(columns=bundle["columns"])
    x = clf.imputer_.transform(X[clf.columns_].values)[0]

    explainer = LimeTabularExplainer(
        bundle["X_train"],
        feature_names=bundle["columns"],
        class_names=list(clf.classes_),
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )
    proba, source = clf.predict_proba_with_source(x.reshape(1, -1))
    top = int(proba[0].argmax())
    exp = explainer.explain_instance(x, clf.predict_proba, num_features=num_features,
                                     labels=(top,), num_samples=num_samples)
    if html_path:
        exp.save_to_file(html_path)

    neighbors = [
        {"uniprot_id": h.target, "label": builder.labels.get(h.target, "?"),
         "name": builder.ref_names.get(h.target, ""), "identity": round(h.identity, 1),
         "evalue": h.evalue, "kind": kind}
        for kind in ("sequence", "structure")
        for h in hits.get(kind, {}).get(protein_id, [])[:5]
    ]
    from .model import reference_match
    known = reference_match(builder, hits["sequence"].get(protein_id, []), sequence)
    return {
        "prediction": known or clf.classes_[top],
        "confidence": 1.0 if known else float(proba[0][top]),
        "method": "reference-match (LIME below explains the model's own call)" if known else source[0],
        "feature_weights": exp.as_list(label=top),
        "neighbors": neighbors,
        "html": html_path,
    }


def format_explanation(result: Dict) -> str:
    lines = [
        f"Prediction: {result['prediction']}  (confidence {result['confidence']:.1%}, via {result['method']})",
        "",
        "Features that pushed toward (+) or away from (-) this prediction:",
    ]
    for cond, w in result["feature_weights"]:
        lines.append(f"  {'+' if w >= 0 else '-'} {abs(w):.3f}  {cond}")
    if result["neighbors"]:
        lines += ["", "Closest annotated proteins:"]
        for n in result["neighbors"]:
            ev = "" if n["evalue"] != n["evalue"] else f", E={n['evalue']:.1e}"  # skip NaN
            name = f" {n['name']}" if n["name"] else ""
            lines.append(f"  [{n['kind']}] {n['uniprot_id']} {n['label']}{name} ({n['identity']}% identity{ev})")
    if result.get("html"):
        lines += ["", f"Interactive LIME chart saved to {result['html']}"]
    return "\n".join(lines)
