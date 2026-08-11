"""hlq.baselines: naive prediction baselines for the funding-timing feasibility check."""
import pytest

from hlq import baselines


def test_majority_class_accuracy_predicts_training_majority():
    # training is 3/4 positive -> majority is 1; test has 2 of 3 positive -> accuracy 2/3
    assert baselines.majority_class_accuracy([1, 1, 1, 0], [1, 0, 1]) == pytest.approx(2 / 3)


def test_majority_class_accuracy_flips_when_training_mostly_negative():
    # training mostly 0 -> majority is 0; test all 0 -> accuracy 1.0
    assert baselines.majority_class_accuracy([0, 0, 0, 1], [0, 0]) == pytest.approx(1.0)


def test_persistence_rmse_matches_hand_value():
    # errors of 0, +2, -2 -> sqrt(mean(0, 4, 4)) = sqrt(8/3)
    assert baselines.persistence_rmse([10.0, 12.0, 8.0], [10.0, 10.0, 10.0]) == pytest.approx((8 / 3) ** 0.5)