from PIL import Image

from ml.evaluate import mask_scores


def test_mask_scores_basic_case():
    pred = Image.new("L", (2, 2), 0)
    true = Image.new("L", (2, 2), 0)
    pred.putpixel((0, 0), 255)
    pred.putpixel((1, 0), 255)
    true.putpixel((0, 0), 255)
    true.putpixel((0, 1), 255)

    scores = mask_scores(pred, true)

    assert scores["mae"] == 0.5
    assert round(scores["iou"], 3) == 0.333
    assert scores["dice"] == 0.5
    assert scores["object_ratio"] == 0.5
    assert scores["fp_rate"] == 0.5
    assert scores["fn_rate"] == 0.5


def test_oxford_trimap_is_converted_to_foreground_and_background():
    pred = Image.new("L", (3, 1), 0)
    pred.putpixel((0, 0), 255)
    pred.putpixel((2, 0), 255)

    trimap = Image.new("L", (3, 1), 2)
    trimap.putpixel((0, 0), 1)
    trimap.putpixel((2, 0), 3)

    scores = mask_scores(pred, trimap)

    assert scores["mae"] == 0.0
    assert scores["iou"] == 1.0
    assert scores["dice"] == 1.0


def test_boundary_error_is_visible_separately():
    true = Image.new("L", (9, 9), 0)
    pred = Image.new("L", (9, 9), 0)
    for y in range(3, 6):
        for x in range(3, 6):
            true.putpixel((x, y), 255)
    for y in range(3, 6):
        for x in range(4, 7):
            pred.putpixel((x, y), 255)

    scores = mask_scores(pred, true)

    assert scores["boundary_mae"] > scores["mae"]
    assert scores["fp_rate"] > 0
    assert scores["fn_rate"] > 0
