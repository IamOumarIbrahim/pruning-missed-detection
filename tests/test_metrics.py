"""Hand-checked cases for the metrics every table depends on."""
import numpy as np

from dms import metrics


def regions(boxes):
    return np.array(boxes, dtype=float).reshape(-1, 1, 4)


def dets(image, conf, cls, box):
    return {"image": np.array(image), "conf": np.array(conf, dtype=float),
            "cls": np.array(cls), "box": np.array(box, dtype=float).reshape(-1, 4)}


BOX = [0, 0, 10, 10]


def test_duplicate_detection_is_a_false_positive():
    gt = [(np.array([0]), regions([BOX]))]
    s = metrics.score_split(dets([0, 0], [0.9, 0.8], [0, 0], [BOX, BOX]), gt)
    m = metrics.at_threshold(s, 0.5)
    assert m["tp"][0] == 1 and m["pred"][0] == 2
    assert m["recall"][0] == 1.0 and m["precision"][0] == 0.5


def test_wrong_class_never_matches():
    gt = [(np.array([1]), regions([BOX]))]
    s = metrics.score_split(dets([0], [0.9], [0], [BOX]), gt)
    assert metrics.at_threshold(s, 0.5)["tp"].sum() == 0


def test_threshold_removes_low_confidence_true_positive():
    gt = [(np.array([0]), regions([BOX]))]
    s = metrics.score_split(dets([0], [0.4], [0], [BOX]), gt)
    assert metrics.at_threshold(s, 0.5)["recall"][0] == 0.0
    assert metrics.at_threshold(s, 0.3)["recall"][0] == 1.0


def test_invisible_box_stays_in_the_denominator():
    gt = [(np.array([3, 3]), regions([BOX, [0, 0, 0, 0]]))]
    s = metrics.score_split(dets([0], [0.9], [3], [BOX]), gt)
    assert s.gt_counts[3] == 2
    assert metrics.at_threshold(s, 0.5)["recall"][3] == 0.5


def test_object_seen_in_two_tiles_counts_once_without_a_false_positive():
    two_tiles = np.array([[[0, 0, 10, 10], [20, 20, 30, 30]]], dtype=float)
    gt = [(np.array([0]), two_tiles)]
    s = metrics.score_split(dets([0, 0], [0.9, 0.8], [0, 0],
                                 [[20, 20, 30, 30], [0, 0, 10, 10]]), gt)
    m = metrics.at_threshold(s, 0.5)
    assert m["tp"][0] == 1 and m["pred"][0] == 2
    assert m["recall"][0] == 1.0 and m["precision"][0] == 1.0


def test_macro_metrics_average_over_classes():
    gt = [(np.array([0, 1]), regions([BOX, [20, 20, 30, 30]]))]
    s = metrics.score_split(dets([0], [0.9], [0], [BOX]), gt)
    m = metrics.at_threshold(s, 0.5)
    assert np.isclose(m["macro_recall"], 0.25)
    assert np.isclose(m["macro_f1"], 0.25)
    assert m["worst_recall"] == 0.0


def test_perfect_ranking_has_unit_average_precision():
    assert np.isclose(metrics.average_precision(np.array([0.5, 1.0]), np.array([1.0, 1.0])), 1.0)


def test_tau_selection_prefers_the_lowest_of_tied_thresholds():
    gt = [(np.array([0]), regions([BOX]))]
    s = metrics.score_split(dets([0], [0.9], [0], [BOX]), gt)
    assert metrics.select_tau(s) == metrics.TAU_GRID[0]
