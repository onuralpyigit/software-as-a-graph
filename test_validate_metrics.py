import pprint
from backend.src.validation.validator import Validator
from backend.src.validation.models import ValidationTargets

def _make_monotonic_data(n: int):
    pred = {f"c{i}": (i + 1) / n for i in range(n - 1)}
    actual = {f"c{i}": (i + 1) / n + 0.01 for i in range(n - 1)}
    pred[f"c{n-1}"] = 3.0
    actual[f"c{n-1}"] = 3.01
    types = {f"c{i}": "App" for i in range(n)}
    return pred, actual, types

pred, actual, types = _make_monotonic_data(20)
validator = Validator(targets=ValidationTargets(
    spearman=0.50,
    spearman_p_max=0.10,
    f1_score=0.30,
    top_5_overlap=0.20,
    rmse_max=0.50,
))
res = validator.validate(pred, actual, types)
# pprint.pprint(res.overall.to_dict())
print("Passed:", res.passed)
print("Spearman:", res.overall.correlation.spearman)
print("Spearman P:", res.overall.correlation.spearman_p)
print("F1:", res.overall.classification.f1_score)
print("Top5:", res.overall.ranking.top_5_overlap)
print("RMSE:", res.overall.error.rmse)
print("Precision:", res.overall.classification.precision)
print("Recall:", res.overall.classification.recall)
import pprint
pred_stats = validator.classifier.compute_stats(list(pred.values()))
actual_stats = validator.classifier.compute_stats(list(actual.values()))

print("Pred Upper:", pred_stats.upper_fence)
print("Actual Upper:", actual_stats.upper_fence)
pred_crit = [v > pred_stats.upper_fence for v in pred.values()]
actual_crit = [v > actual_stats.upper_fence for v in actual.values()]
print("Pred Crit:", sum(pred_crit))
print("Actual Crit:", sum(actual_crit))
from backend.src.validation.metric_calculator import calculate_classification
c = calculate_classification(pred_crit, actual_crit)
print("TP:", c.true_positives, "FP:", c.false_positives, "FN:", c.false_negatives)
import pprint
from backend.src.validation.validator import Validator
from backend.src.validation.models import ValidationTargets

def _make_monotonic_data(n: int):
    pred = {f"c{i}": (i + 1) / n for i in range(n - 1)}
    actual = {f"c{i}": (i + 1) / n + 0.01 for i in range(n - 1)}
    pred[f"c{n-1}"] = 3.0
    actual[f"c{n-1}"] = 3.01
    types = {f"c{i}": "App" for i in range(n)}
    return pred, actual, types

pred, actual, types = _make_monotonic_data(20)
validator = Validator(targets=ValidationTargets(), winsorize_actuals=False)
res = validator.validate(pred, actual, types)
print("Without winsorize:")
print("Passed:", res.passed)
print("F1:", res.overall.classification.f1_score)
print("TP:", res.overall.classification.true_positives)

validator2 = Validator(targets=ValidationTargets(), winsorize_actuals=True)
res2 = validator2.validate(pred, actual, types)
print("With winsorize:")
print("Passed:", res2.passed)
print("F1:", res2.overall.classification.f1_score)
print("TP:", res2.overall.classification.true_positives)
