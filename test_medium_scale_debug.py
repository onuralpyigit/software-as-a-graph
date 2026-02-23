import sys
from backend.src.validation.validator import Validator

n = 50
pred = {f"c{i}": (i + 1) / n for i in range(n - 1)}
pred[f"c{n-1}"] = 5.0  # outlier
actual = {f"c{i}": (i + 1) / n + 0.005 for i in range(n - 1)}
actual[f"c{n-1}"] = 5.01  # outlier
types = {f"c{i}": "Application" for i in range(n)}

validator = Validator(winsorize_actuals=False)
res = validator.validate(pred, actual, types)
ids = list(pred.keys())
pred_vals = [pred[k] for k in ids]
pred_stats = validator.classifier.compute_stats(pred_vals)
actual_vals = [actual[k] for k in ids]
actual_stats = validator.classifier.compute_stats(actual_vals)
print(f"Pred upper fence: {pred_stats.upper_fence}")
print(f"Pred outliers: {[v for v in pred_vals if v > pred_stats.upper_fence]}")
print(f"Actual upper fence: {actual_stats.upper_fence}")
print(f"Actual outliers: {[v for v in actual_vals if v > actual_stats.upper_fence]}")
print("Classification:", res.overall.classification)
