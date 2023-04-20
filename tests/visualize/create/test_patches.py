import pytest

from visualize.create.patches import (
    ColorRgb,
    _bbox_color,
    RED_RGB,
    GREEN_RGB,
    YELLOW_RGB,
)


@pytest.mark.parametrize(
    "similarity,expected_color",
    [
        (0.0, RED_RGB),
        (0.2, (255, 102, 0)),
        (0.5, YELLOW_RGB),
        (0.6, (204, 255, 0)),
        (0.9, (50, 255, 0)),
        (1.0, GREEN_RGB),
    ],
)
def test_bbox_color_happy(similarity: float, expected_color: ColorRgb):
    # TODO: It's a bit odd testing internal/protected functions, but it's a bit tricky to test the higher level image
    #  behaviour with the current code.
    actual_color = _bbox_color(similarity)
    assert actual_color == expected_color
