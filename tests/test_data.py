"""Input transforms: boxes must land where the pixels came from."""
import numpy as np

from dms import data


def test_crop_round_trip_reaches_the_frame():
    dets = {"image": np.array([0]), "conf": np.array([0.9]), "cls": np.array([0]),
            "box": np.array([[0.0, 0.0, 1.0, 1.0]])}
    assert data.to_frame(dets, "cropped")["box"].tolist() == [[140, 120, 540, 520]]


def test_clip_drops_boxes_outside_the_window():
    boxes = np.array([[0.0, 0.0, 100.0, 100.0], [150.0, 150.0, 200.0, 200.0]])
    out = data._clip(boxes, (140, 120, 540, 520))
    assert out[0].tolist() == [0, 0, 0, 0]
    assert out[1].tolist() == [150, 150, 200, 200]


def test_collage_box_returns_to_its_source_tile():
    # A detection in the head strip (left half of the canvas) keeps y, and shifts x by 173.
    box = np.array([[100.0, 100.0, 200.0, 200.0]]) / 640
    dets = {"image": np.array([0]), "conf": np.array([0.9]), "cls": np.array([0]), "box": box}
    assert data.to_frame(dets, "collage")["box"].round().tolist() == [[273, 100, 373, 200]]
