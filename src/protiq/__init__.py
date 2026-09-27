"""ProtiQ: automated protein function prediction from sequence and structure similarity.

Pipeline: sequence search (BLAST-style) + structure search (Foldseek on AlphaFold models)
-> neighbor votes and similarity scores -> k-nearest neighbors classifier
(with a logistic regression fallback) -> LIME explanations.
"""

__version__ = "0.2.0"
__authors__ = ["Muyi Tang", "Xenia Huang"]
