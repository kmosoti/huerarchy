"""Trusted evaluation components. Clean labels never travel back to the learner."""
import numpy as np

from .types import register


@register("evaluator", "heldout-classification", "Balanced accuracy, proper scoring rules, and confidence diagnostics")
class HeldOutEvaluator:
    def evaluate(self, probabilities, labels):
        if (probabilities.shape != labels.shape or not np.all(np.isfinite(probabilities))
                or np.any(probabilities < 0) or np.any(probabilities > 1)):
            raise ValueError("invalid evaluated probabilities")
        prediction = probabilities >= 0.5
        positive = labels.astype(bool)
        if not np.any(positive) or np.all(positive):
            raise ValueError("balanced evaluation requires both classes")
        tpr = float(np.mean(prediction[positive]))
        tnr = float(np.mean(~prediction[~positive]))
        clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
        confidence = np.maximum(probabilities, 1 - probabilities) >= 0.9
        return {"balanced_accuracy": (tpr + tnr) / 2,
                "brier": float(np.mean((probabilities - labels) ** 2)),
                "log_loss": float(-np.mean(np.where(positive, np.log(clipped), np.log1p(-clipped)))),
                "confident_coverage": float(np.mean(confidence)),
                "confident_errors": int(np.sum(confidence & (prediction != positive))),
                "confident_count": int(np.sum(confidence))}
