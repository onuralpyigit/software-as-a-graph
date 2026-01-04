import pytest
from src.validation.metrics import (
    calculate_classification, 
    spearman_correlation, 
    calculate_ranking_metrics
)
from src.validation.validator import Validator

def test_metrics_logic():
    # Perfect correlation
    x = [0.1, 0.2, 0.3, 0.4, 0.5]
    y = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert spearman_correlation(x, y) == 1.0
    
    # Inverse correlation
    y_inv = [0.5, 0.4, 0.3, 0.2, 0.1]
    assert spearman_correlation(x, y_inv) == -1.0

def test_ranking_logic():
    # Pred: A is highest. Actual: A is highest.
    pred = {"A": 0.9, "B": 0.5, "C": 0.1}
    act  = {"A": 0.8, "B": 0.4, "C": 0.2}
    
    res = calculate_ranking_metrics(pred, act)
    assert res.top_5_overlap == 1.0 # Sets match exactly
    assert res.ndcg_10 > 0.9 # High quality ranking

def test_validator_flow():
    validator = Validator()
    pred = {"A": 0.9, "B": 0.1, "C": 0.5}
    act  = {"A": 0.8, "B": 0.2, "C": 0.6}
    types = {"A": "Type1", "B": "Type1", "C": "Type2"}
    
    res = validator.validate(pred, act, types)
    
    assert res.overall.sample_size == 3
    assert "Type1" in res.by_type
    assert res.by_type["Type1"].sample_size == 2