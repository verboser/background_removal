import subprocess
import sys
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATE = PROJECT_ROOT / "ml" / "evaluate.py"


def test_evaluate_reports_missing_image_dir(tmp_path):
    masks = tmp_path / "masks"
    masks.mkdir()

    result = subprocess.run(
        [sys.executable, str(EVALUATE), "--images", str(tmp_path / "nope"), "--masks", str(masks)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Папка с картинками не найдена" in result.stderr


def test_evaluate_reports_empty_images(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()

    result = subprocess.run(
        [sys.executable, str(EVALUATE), "--images", str(images), "--masks", str(masks)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Картинки не найдены" in result.stderr


def test_evaluate_checks_pairs_before_model_loading(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    Image.new("RGB", (4, 4), "white").save(images / "sample.jpg")

    result = subprocess.run(
        [sys.executable, str(EVALUATE), "--images", str(images), "--masks", str(masks)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "нет маски для sample.jpg" in result.stdout
    assert "Не нашлось ни одной пары" in result.stderr


def test_evaluate_strict_file_list_reports_missing_images(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    file_list = tmp_path / "files.txt"
    images.mkdir()
    masks.mkdir()
    file_list.write_text("missing_sample\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(EVALUATE),
            "--images",
            str(images),
            "--masks",
            str(masks),
            "--file-list",
            str(file_list),
            "--strict",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Нет картинок для 1 строк" in result.stderr
