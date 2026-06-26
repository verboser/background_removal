import argparse
import csv
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import Settings
from backend.models.loader import load_background_removal_model


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def mask_scores(
    predicted: Image.Image, target: Image.Image, threshold: float = 0.5
) -> dict[str, float]:
    if predicted.size != target.size:
        predicted = predicted.resize(target.size, Image.Resampling.LANCZOS)

    pred = np.asarray(predicted.convert("L"), dtype=np.float32) / 255.0
    target_values = np.asarray(target.convert("L"))
    if target_values.max() <= 3 and target_values.min() >= 1:
        # В Oxford Pets class 3 это неопределенный край. Для удаления фона
        # я отношу его к объекту, иначе шерсть и лапы часто режутся слишком жестко.
        true = np.isin(target_values, (1, 3)).astype(np.float32)
    else:
        true = target_values.astype(np.float32) / 255.0

    pred_bin = pred >= threshold
    true_bin = true >= threshold

    intersection = np.logical_and(pred_bin, true_bin).sum()
    union = np.logical_or(pred_bin, true_bin).sum()
    pred_area = pred_bin.sum()
    true_area = true_bin.sum()
    background_area = pred_bin.size - true_area
    false_positive = np.logical_and(pred_bin, ~true_bin).sum()
    false_negative = np.logical_and(~pred_bin, true_bin).sum()

    # Это не matting-метрика из статьи, а быстрый способ отдельно увидеть боль по краю.
    mae_map = np.abs(pred - true)

    return {
        "mae": float(mae_map.mean()),
        "iou": float(intersection / union) if union else 1.0,
        "dice": float((2 * intersection) / (pred_area + true_area))
        if pred_area + true_area
        else 1.0,
        "object_ratio": float(true_area / pred_bin.size),
        "fp_rate": float(false_positive / background_area) if background_area else 0.0,
        "fn_rate": float(false_negative / true_area) if true_area else 0.0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--backend", choices=["rembg", "birefnet"], default="rembg")
    parser.add_argument("--out", type=Path, default=Path("outputs/metrics.csv"))
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--gallery", type=Path)
    parser.add_argument("--gallery-items", type=int, default=6)
    parser.add_argument("--file-list", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rembg-model", default="birefnet-general")
    parser.add_argument("--birefnet-model", default="ZhengPeng7/BiRefNet")
    parser.add_argument("--birefnet-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if not args.images.is_dir():
        raise SystemExit(f"Папка с картинками не найдена: {args.images}")
    if not args.masks.is_dir():
        raise SystemExit(f"Папка с масками не найдена: {args.masks}")

    if args.file_list:
        if not args.file_list.is_file():
            raise SystemExit(f"Список файлов не найден: {args.file_list}")
        wanted = [
            line.strip().split()[0]
            for line in args.file_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        image_paths = []
        missing_images = []
        for name in wanted:
            image_path = None
            stem = Path(name).stem
            for suffix in sorted(IMAGE_SUFFIXES):
                candidate = args.images / f"{stem}{suffix}"
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path is None:
                missing_images.append(name)
                continue
            image_paths.append(image_path)
        if args.strict and missing_images:
            sample = ", ".join(missing_images[:5])
            raise SystemExit(f"Нет картинок для {len(missing_images)} строк: {sample}")
    else:
        missing_images = []
        image_paths = [
            path
            for path in sorted(args.images.iterdir())
            if path.suffix.lower() in IMAGE_SUFFIXES
        ]
    if args.limit:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise SystemExit(f"Картинки не найдены в {args.images}")

    pairs = []
    missing_masks = []
    for image_path in image_paths:
        mask_path = None
        for suffix in sorted(IMAGE_SUFFIXES):
            candidate = args.masks / f"{image_path.stem}{suffix}"
            if candidate.exists():
                mask_path = candidate
                break
        if mask_path is None:
            missing_masks.append(image_path.name)
            print(f"нет маски для {image_path.name}")
            continue
        pairs.append((image_path, mask_path))

    if args.strict and missing_masks:
        sample = ", ".join(missing_masks[:5])
        raise SystemExit(f"Нет масок для {len(missing_masks)} файлов: {sample}")
    if not pairs:
        raise SystemExit("Не нашлось ни одной пары картинка + маска")

    settings = Settings(
        model_backend=args.backend,
        rembg_model=args.rembg_model,
        birefnet_model_id=args.birefnet_model,
        birefnet_image_size=args.birefnet_size,
        device=args.device,
    )
    model = load_background_removal_model(settings)
    rows = []
    gallery_rows = []

    for image_path, mask_path in pairs:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        with Image.open(mask_path) as source:
            target = source.convert("L")

        started = perf_counter()
        alpha = model["predict"](image)
        latency_ms = round((perf_counter() - started) * 1000, 1)
        scores = mask_scores(alpha, target, threshold=args.threshold)
        if args.gallery:
            gallery_rows.append((image_path, image.copy(), target.copy(), alpha.copy()))

        rows.append(
            {
                "file": image_path.name,
                "backend": args.backend,
                "model": model["name"],
                "width": image.width,
                "height": image.height,
                "latency_ms": latency_ms,
                "mae": round(scores["mae"], 5),
                "iou": round(scores["iou"], 5),
                "dice": round(scores["dice"], 5),
                "object_ratio": round(scores["object_ratio"], 5),
                "fp_rate": round(scores["fp_rate"], 5),
                "fn_rate": round(scores["fn_rate"], 5),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    mae = np.mean([row["mae"] for row in rows])
    iou = np.mean([row["iou"] for row in rows])
    dice = np.mean([row["dice"] for row in rows])
    latency = np.mean([row["latency_ms"] for row in rows])
    latency_p50 = np.percentile([row["latency_ms"] for row in rows], 50)
    latency_p90 = np.percentile([row["latency_ms"] for row in rows], 90)
    worst_count = max(5, args.gallery_items if args.gallery else 0)
    worst_mae = sorted(rows, key=lambda row: row["mae"], reverse=True)[:worst_count]
    worst_boundary = sorted(rows, key=lambda row: row["boundary_mae"], reverse=True)[:worst_count]

    buckets = {
        "малый объект (<25%)": [row for row in rows if row["object_ratio"] < 0.25],
        "средний объект (25-55%)": [
            row for row in rows if 0.25 <= row["object_ratio"] < 0.55
        ],
        "крупный объект (>=55%)": [row for row in rows if row["object_ratio"] >= 0.55],
    }

    summary_path = args.summary or args.out.with_suffix(".summary.md")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file:
        file.write("# Оценка масок\n\n")
        file.write(f"Модель: `{model['name']}` через `{args.backend}`\n\n")
        file.write("## Протокол\n\n")
        file.write(f"- картинки: `{args.images}`\n")
        file.write(f"- маски: `{args.masks}`\n")
        if args.file_list:
            file.write(f"- список файлов: `{args.file_list}`\n")
        file.write(f"- найдено пар: {len(rows)}\n")
        file.write(f"- без картинки: {len(missing_images)}\n")
        file.write(f"- без маски: {len(missing_masks)}\n")
        file.write(f"- порог бинаризации: {args.threshold}\n")
        file.write("- для Oxford Pets trimap класс 2 считается фоном, 1 и 3 объектом\n\n")
        file.write("## Средние значения\n\n")
        file.write("| MAE | IoU | Dice | latency avg | latency p50 | latency p90 |\n")
        file.write("| ---: | ---: | ---: | ---: | ---: | ---: |\n")
        file.write(
            f"| {mae:.5f} | {iou:.5f} | {dice:.5f} | "
            f"{latency:.1f} ms | {latency_p50:.1f} ms | {latency_p90:.1f} ms |\n\n"
        )
        file.write("## Размер объекта\n\n")
        file.write("| Группа | count | MAE | IoU | Dice |\n")
        file.write("| --- | ---: | ---: | ---: | ---: |\n")
        for name, bucket_rows in buckets.items():
            if not bucket_rows:
                file.write(f"| {name} | 0 | - | - | - |\n")
                continue
            file.write(
                f"| {name} | {len(bucket_rows)} | "
                f"{np.mean([row['mae'] for row in bucket_rows]):.5f} | "
                f"{np.mean([row['iou'] for row in bucket_rows]):.5f} | "
                f"{np.mean([row['dice'] for row in bucket_rows]):.5f} |\n"
            )
        file.write("\n## Худшие случаи по MAE\n\n")
        file.write("| Файл | MAE | IoU | Dice | FP rate | FN rate |\n")
        file.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
        for row in worst_mae:
            file.write(
                f"| {row['file']} | {row['mae']:.5f} | {row['iou']:.5f} | "
                f"{row['dice']:.5f} | {row['fp_rate']:.5f} | {row['fn_rate']:.5f} |\n"
            )
        file.write("\n## Худшие края\n\n")
        file.write("| Файл | MAE | IoU | Dice |\n")
        file.write("| --- | ---: | ---: | ---: | ---: |\n")
        for row in worst_boundary:
            file.write(
                f"| {row['file']} | {row['boundary_mae']:.5f} | {row['mae']:.5f} | "
                f"{row['iou']:.5f} | {row['dice']:.5f} |\n"
            )
        file.write("\n## По времени\n\n")
        file.write(
            "Это не нагрузочный тест, но по CPU-прогону видно верхнюю оценку для "
            f"последовательной обработки: около {1000 / latency:.2f} изображения/сек.\n"
        )

    if args.gallery:
        args.gallery.parent.mkdir(parents=True, exist_ok=True)
        selected = {row["file"] for row in worst_mae[: args.gallery_items]}
        preview_rows = [item for item in gallery_rows if item[0].name in selected]
        cell_w, cell_h = 220, 190
        header_h = 34
        sheet = Image.new(
            "RGB",
            (cell_w * 4, header_h + cell_h * len(preview_rows)),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for i, title in enumerate(("source", "target", "prediction", "error")):
            draw.text((i * cell_w + 8, 10), title, fill=(20, 20, 20))

        for row_index, (image_path, image, target, alpha) in enumerate(preview_rows):
            y0 = header_h + row_index * cell_h
            target_values = np.asarray(target.convert("L"))
            if target_values.max() <= 3 and target_values.min() >= 1:
                true_alpha = Image.fromarray(
                    (np.isin(target_values, (1, 3)).astype(np.uint8) * 255),
                    mode="L",
                )
            else:
                true_alpha = target.convert("L")

            pred_alpha = alpha.convert("L")
            if pred_alpha.size != image.size:
                pred_alpha = pred_alpha.resize(image.size, Image.Resampling.LANCZOS)
            if true_alpha.size != image.size:
                true_alpha = true_alpha.resize(image.size, Image.Resampling.NEAREST)

            checker = Image.new("RGB", image.size, (235, 235, 235))
            checker_draw = ImageDraw.Draw(checker)
            step = 16
            for yy in range(0, image.height, step):
                for xx in range(0, image.width, step):
                    if (xx // step + yy // step) % 2:
                        checker_draw.rectangle(
                            (xx, yy, xx + step - 1, yy + step - 1),
                            fill=(205, 205, 205),
                        )

            target_cut = Image.composite(image, checker, true_alpha)
            pred_cut = Image.composite(image, checker, pred_alpha)
            pred_arr = np.asarray(pred_alpha, dtype=np.int16)
            true_arr = np.asarray(true_alpha, dtype=np.int16)
            error = np.abs(pred_arr - true_arr).astype(np.uint8)
            # Зеленый фон оставил намеренно: на нем красные ошибки быстрее считываются глазами.
            error_rgb = np.zeros((error.shape[0], error.shape[1], 3), dtype=np.uint8)
            error_rgb[:, :, 0] = error
            error_rgb[:, :, 1] = 255 - error
            error_rgb[:, :, 2] = 60
            error_image = Image.fromarray(error_rgb, mode="RGB")

            for col, item in enumerate((image, target_cut, pred_cut, error_image)):
                item = item.copy()
                item.thumbnail((cell_w - 20, cell_h - 36), Image.Resampling.LANCZOS)
                x = col * cell_w + (cell_w - item.width) // 2
                y = y0 + 28 + (cell_h - 36 - item.height) // 2
                sheet.paste(item.convert("RGB"), (x, y))
            draw.text((8, y0 + 8), image_path.name, fill=(20, 20, 20))
        sheet.save(args.gallery, quality=92)
        with summary_path.open("a", encoding="utf-8") as file:
            file.write(f"\nГалерея худших случаев: `{args.gallery}`\n")

    print(
        f"{args.backend}: files={len(rows)}, skipped={len(missing_masks)}, "
        f"mae={mae:.5f}, iou={iou:.5f}, dice={dice:.5f}, "
        f"latency_ms={latency:.1f}"
    )
    print(f"csv: {args.out}")
    print(f"summary: {summary_path}")
    if args.gallery:
        print(f"gallery: {args.gallery}")
