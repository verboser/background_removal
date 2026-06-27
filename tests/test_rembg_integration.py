from pathlib import Path

import pytest
from PIL import Image

from backend.config import Settings
from backend.models.loader import load_background_removal_model
from ml.evaluate import mask_scores


pytestmark = pytest.mark.integration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = PROJECT_ROOT / "data/oxford_pets/images_eval50/chihuahua_191.jpg"
MASK_PATH = PROJECT_ROOT / "data/oxford_pets/annotations/trimaps/chihuahua_191.png"


def test_rembg_on_oxford_pet():
    if not IMAGE_PATH.exists() or not MASK_PATH.exists():
        pytest.skip("Нет Oxford Pets eval50 с trimap-масками")

    model = load_background_removal_model(
        Settings(model_backend="rembg", rembg_model="birefnet-general")
    )
    with Image.open(IMAGE_PATH) as source:
        image = source.convert("RGB")
    with Image.open(MASK_PATH) as source:
        target = source.convert("L")

    alpha = model["predict"](image)
    scores = mask_scores(alpha, target)

    assert alpha.mode == "L"
    assert alpha.size == image.size
    assert scores["iou"] > 0.6
    assert scores["dice"] > 0.75


def test_birefnet_on_oxford_pet():
    if not IMAGE_PATH.exists() or not MASK_PATH.exists():
        pytest.skip("Нет Oxford Pets eval50 с trimap-масками")

    model = load_background_removal_model(
        Settings(model_backend="birefnet", birefnet_image_size=512, device="cpu")
    )
    with Image.open(IMAGE_PATH) as source:
        image = source.convert("RGB")
    with Image.open(MASK_PATH) as source:
        target = source.convert("L")

    alpha = model["predict"](image)
    scores = mask_scores(alpha, target)

    assert alpha.mode == "L"
    assert alpha.size == image.size
    assert scores["iou"] > 0.6
    assert scores["dice"] > 0.75
