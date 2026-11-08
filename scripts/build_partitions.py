"""Build the input partitions of Phase II from the 640 x 640 frames.

data/partitions/<method>/<res>x<res>/<split>/{images,labels}
Crop and collage boxes come from configs/experiment.yaml. Resizing is bilinear. Labels are
clipped to what stays visible; the scoring step keeps the removed boxes in its denominator.
"""
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dms.data import MIN_VISIBLE, frame_path, frame_ground_truth, split_stems  # noqa: E402
from dms.paths import CFG, PARTITIONS  # noqa: E402

INPUT = CFG["input"]


def windows(method):
    """(source window, destination window) pairs on a 640 x 640 canvas."""
    if method == "cropped":
        return [(INPUT["crop_box"], (0, 0, 640, 640))]
    if method == "collage":
        return [(t["src"], t["dst"]) for t in INPUT["collage_tiles"].values()]
    return [((0, 0, 640, 640), (0, 0, 640, 640))]


def render(image, method):
    if method == "baseline":
        return image
    if method == "cropped":
        return image.crop(INPUT["crop_box"])
    canvas = Image.new("RGB", (640, 640))
    for src, dst in windows(method):
        canvas.paste(image.crop(src), dst[:2])
    return canvas


def labels(classes, boxes, method):
    lines = []
    for cls, (x1, y1, x2, y2) in zip(classes, boxes):
        for (sx1, sy1, sx2, sy2), (dx1, dy1, dx2, dy2) in windows(method):
            ix1, iy1, ix2, iy2 = max(x1, sx1), max(y1, sy1), min(x2, sx2), min(y2, sy2)
            if ix2 - ix1 < MIN_VISIBLE or iy2 - iy1 < MIN_VISIBLE:
                continue
            kx, ky = (dx2 - dx1) / (sx2 - sx1), (dy2 - dy1) / (sy2 - sy1)
            nx1, ny1 = dx1 + (ix1 - sx1) * kx, dy1 + (iy1 - sy1) * ky
            nx2, ny2 = dx1 + (ix2 - sx1) * kx, dy1 + (iy2 - sy1) * ky
            lines.append(f"{cls} {(nx1 + nx2) / 1280:.6f} {(ny1 + ny2) / 1280:.6f} "
                         f"{(nx2 - nx1) / 640:.6f} {(ny2 - ny1) / 640:.6f}")
    return lines


def main():
    for method in ("baseline", "cropped", "collage"):
        for res in INPUT["resolutions"]:
            for split in ("train", "val", "test"):
                root = PARTITIONS / method / f"{res}x{res}" / split
                (root / "images").mkdir(parents=True, exist_ok=True)
                (root / "labels").mkdir(parents=True, exist_ok=True)
                truth = frame_ground_truth(split)
                for stem, (classes, boxes) in zip(split_stems(split), truth):
                    image = render(Image.open(frame_path(stem)).convert("RGB"), method)
                    if image.size != (res, res):
                        image = image.resize((res, res), Image.Resampling.BILINEAR)
                    image.save(root / "images" / f"{stem}.jpg", quality=95)
                    (root / "labels" / f"{stem}.txt").write_text(
                        "\n".join(labels(classes, boxes, method)), encoding="utf-8")
            print(f"{method} {res}: done", flush=True)


if __name__ == "__main__":
    main()
