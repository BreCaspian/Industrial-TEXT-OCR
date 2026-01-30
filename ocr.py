import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

def _preparse_flags(argv: list) -> dict:
    return {
        "disable_pir": "--disable_pir" in argv,
        "disable_mkldnn": "--disable_mkldnn" in argv,
        "disable_new_executor": "--disable_new_executor" in argv,
    }


_flags = _preparse_flags(sys.argv)
if _flags["disable_pir"]:
    os.environ["PADDLE_DISABLE_PIR"] = "1"
    os.environ["FLAGS_enable_pir_api"] = "0"
    os.environ["FLAGS_use_pir_api"] = "0"
if _flags["disable_mkldnn"]:
    os.environ["FLAGS_use_mkldnn"] = "0"
if _flags["disable_new_executor"]:
    os.environ["FLAGS_new_executor"] = "0"

import cv2
import numpy as np
import pandas as pd


@dataclass
class OCRResult:
    best_code: str
    raw_text: str
    confidence: float
    chosen_rotation: int
    best_box: Optional[np.ndarray]
    all_codes: List[str]
    ocr_items: List[dict]


def imread_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: str, img) -> bool:
    ext = os.path.splitext(path)[1].lower()
    if not ext:
        return False
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(path)
    return True


def resize_max_side(img, max_side: int):
    if max_side <= 0:
        return img
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def rotate_image(img, rot: int):
    rot = rot % 360
    if rot == 0:
        return img
    if rot == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rot == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if rot == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def preprocess(img, mode: str):
    if mode == "none":
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode == "mild":
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if mode == "bin":
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    return img


def overlay_info(img, text: str):
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(1.0, w / 1200.0))
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 6
    x, y = 8, 8 + th
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + pad), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)
    return img


def normalize_token(text: str) -> str:
    return re.sub(r"\s+", "", text).upper()


def run_ocr_paddle(ocr, img):
    try:
        res = ocr.ocr(img, cls=True)
    except TypeError:
        res = ocr.ocr(img)
    items = []
    if res and res[0]:
        for item in res[0]:
            box = np.array(item[0], dtype=np.int32)
            text, conf = item[1][0], float(item[1][1])
            items.append({"box": box, "text": text, "conf": conf})
    full = " ".join([i["text"] for i in items])
    conf = float(sum(i["conf"] for i in items) / len(items)) if items else 0.0
    return items, full, conf


def run_ocr_easyocr(reader, img):
    results = reader.readtext(img)
    items = []
    for box, text, conf in results:
        box_np = np.array(box, dtype=np.int32)
        items.append({"box": box_np, "text": text, "conf": float(conf)})
    full = " ".join([i["text"] for i in items])
    conf = float(sum(i["conf"] for i in items) / len(items)) if items else 0.0
    return items, full, conf


def find_best_code(
    items: List[dict], regex: Optional[str], min_len: int
) -> Tuple[str, Optional[np.ndarray], List[str]]:
    tokens = [normalize_token(i["text"]) for i in items]
    all_codes = []
    best = ("", None, 0.0, -1)
    max_span = 4
    for i in range(len(tokens)):
        if not tokens[i]:
            continue
        for j in range(i, min(len(tokens), i + max_span)):
            combined = "".join(tokens[i : j + 1])
            if len(combined) < min_len:
                continue
            if regex:
                if not re.fullmatch(regex, combined):
                    continue
            else:
                if not re.fullmatch(r"[A-Z0-9]+", combined):
                    continue
            all_codes.append(combined)
            conf = sum(items[k]["conf"] for k in range(i, j + 1)) / (j - i + 1)
            if (len(combined), conf) > (len(best[0]), best[2]):
                boxes = [items[k]["box"] for k in range(i, j + 1)]
                merged = np.concatenate(boxes, axis=0)
                x_min = int(np.min(merged[:, 0]))
                y_min = int(np.min(merged[:, 1]))
                x_max = int(np.max(merged[:, 0]))
                y_max = int(np.max(merged[:, 1]))
                rect = np.array(
                    [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
                    dtype=np.int32,
                )
                best = (combined, rect, conf, j - i + 1)
    return best[0], best[1], list(dict.fromkeys(all_codes))


def best_of_rotations(
    ocr,
    backend: str,
    img,
    regex: Optional[str],
    min_len: int,
    do_rotate: bool,
):
    rotations = [0, 90, 180, 270] if do_rotate else [0]
    best = None
    best_key = None
    for rot in rotations:
        img_r = rotate_image(img, rot)
        if backend == "easyocr":
            items, full, conf = run_ocr_easyocr(ocr, img_r)
        else:
            items, full, conf = run_ocr_paddle(ocr, img_r)
        code, best_box, all_codes = find_best_code(items, regex, min_len)
        key = (1 if code else 0, len(code), conf)
        if best is None or key > best_key:
            best = OCRResult(code, full, conf, rot, best_box, all_codes, items)
            best_key = key
    return best


def list_images(folder: str, exts: Tuple[str, ...]):
    items = []
    for fn in os.listdir(folder):
        if fn.lower().endswith(exts):
            items.append(os.path.join(folder, fn))
    items.sort()
    return items


def parse_args():
    p = argparse.ArgumentParser(
        description="Batch OCR for alphanumeric codes with rotation handling."
    )
    p.add_argument("-i", "--input", default="imgs", help="Input image folder")
    p.add_argument("-o", "--output", default="output", help="Output folder")
    p.add_argument("--csv_name", default="ocr_results.csv", help="CSV file name")
    p.add_argument("--min_conf", type=float, default=0.75, help="Min confidence threshold")
    p.add_argument("--min_len", type=int, default=5, help="Min code length")
    p.add_argument("--regex", default=None, help="Custom regex for code extraction")
    p.add_argument("--preprocess", choices=["none", "mild", "bin"], default="mild")
    p.add_argument("--max_side", type=int, default=1600, help="Resize max side")
    p.add_argument("--no_rotate", action="store_true", help="Disable 0/90/180/270 search")
    p.add_argument("--save_review", action="store_true", help="Save review images")
    p.add_argument("--save_annotated", action="store_true", help="Save annotated images")
    p.add_argument("--disable_pir", action="store_true", help="Set PADDLE_DISABLE_PIR=1")
    p.add_argument("--disable_mkldnn", action="store_true", help="Set FLAGS_use_mkldnn=0")
    p.add_argument("--disable_new_executor", action="store_true", help="Set FLAGS_new_executor=0")
    p.add_argument(
        "--backend",
        choices=["paddle", "easyocr"],
        default="paddle",
        help="OCR backend",
    )
    p.add_argument("--use_gpu", action="store_true", help="Enable GPU")
    p.add_argument("--gpu_id", default="0", help="CUDA device id (default 0)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if args.disable_pir:
        os.environ["PADDLE_DISABLE_PIR"] = "1"
        os.environ["FLAGS_enable_pir_api"] = "0"
        os.environ["FLAGS_use_pir_api"] = "0"
    if args.disable_mkldnn:
        os.environ["FLAGS_use_mkldnn"] = "0"
    if args.disable_new_executor:
        os.environ["FLAGS_new_executor"] = "0"

    backend = args.backend
    if backend == "easyocr":
        try:
            import easyocr
        except Exception as e:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "EasyOCR is not available. Install with: pip install easyocr"
            ) from e
        ocr = easyocr.Reader(["en"], gpu=args.use_gpu)
    else:
        try:
            from paddleocr import PaddleOCR
        except Exception as e:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "PaddleOCR is not available. Install with: pip install paddleocr opencv-python pandas"
            ) from e
        try:
            ocr = PaddleOCR(use_textline_orientation=True, lang="en", use_gpu=args.use_gpu)
        except TypeError:
            ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=args.use_gpu)

    img_dir = args.input
    out_dir = args.output
    review_dir = os.path.join(out_dir, "review")
    os.makedirs(out_dir, exist_ok=True)
    if args.save_review:
        os.makedirs(review_dir, exist_ok=True)
    annotated_dir = os.path.join(out_dir, "annotated")
    if args.save_annotated:
        os.makedirs(annotated_dir, exist_ok=True)

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    img_paths = list_images(img_dir, exts)
    if not img_paths:
        print(f"No images found in: {img_dir}")
        return

    rows = []
    need_check_count = 0

    for idx, path in enumerate(img_paths, 1):
        fn = os.path.basename(path)
        img0 = imread_unicode(path)
        if img0 is None:
            rows.append(
                {
                    "file": fn,
                    "best_code": "",
                    "raw_text": "",
                    "confidence": 0.0,
                    "chosen_rotation": 0,
                    "need_manual_check": True,
                    "status": "read_fail",
                }
            )
            need_check_count += 1
            print(f"[{idx}/{len(img_paths)}] {fn} -> read_fail  [CHECK]")
            continue

        img0 = resize_max_side(img0, args.max_side)
        img0 = preprocess(img0, args.preprocess)
        result = best_of_rotations(
            ocr, backend, img0, args.regex, args.min_len, not args.no_rotate
        )

        need_check = (result.confidence < args.min_conf) or (result.best_code == "")
        if need_check:
            need_check_count += 1

        if args.save_annotated or (args.save_review and need_check):
            imgR = rotate_image(img0, result.chosen_rotation)
            annotated = imgR.copy()
            for it in result.ocr_items:
                box = it["box"].astype(int)
                cv2.polylines(annotated, [box], True, (0, 255, 0), 2)
                tl = box[0]
                label = normalize_token(it["text"])
                cv2.putText(
                    annotated,
                    label,
                    (int(tl[0]), int(tl[1]) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            if result.best_box is not None:
                cv2.polylines(
                    annotated,
                    [result.best_box.astype(int)],
                    True,
                    (0, 0, 255),
                    3,
                )
            label = (
                f"{fn} | rot={result.chosen_rotation} | "
                f"conf={result.confidence:.2f} | {result.best_code}"
            )
            annotated = overlay_info(annotated, label)

            if args.save_annotated:
                out_path = os.path.join(annotated_dir, fn)
                imwrite_unicode(out_path, annotated)
            if args.save_review and need_check:
                out_path = os.path.join(review_dir, fn)
                imwrite_unicode(out_path, annotated)

        rows.append(
            {
                "file": fn,
                "best_code": result.best_code,
                "all_codes": "|".join(result.all_codes),
                "raw_text": re.sub(r"\s+", " ", result.raw_text).strip(),
                "confidence": round(result.confidence, 4),
                "chosen_rotation": result.chosen_rotation,
                "need_manual_check": need_check,
                "status": "ok",
            }
        )

        print(
            f"[{idx}/{len(img_paths)}] {fn} -> code={result.best_code} "
            f"conf={result.confidence:.3f} rot={result.chosen_rotation}"
            + ("  [CHECK]" if need_check else "")
        )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, args.csv_name)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nDone. CSV saved to: {csv_path}")
    print(f"Total: {len(df)} | Need manual check: {need_check_count}")
    if args.save_review:
        print(f"Review images saved to: {review_dir}")


if __name__ == "__main__":
    main()
