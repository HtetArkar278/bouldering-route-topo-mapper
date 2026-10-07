"""
Stage 4: classify each detected hold's color.

Detection (hold_detection.py) finds WHERE holds are without knowing
their colors. This script determines WHAT color each one is, as a
genuinely separate step -- unlike Milestone 1's synthetic baseline,
where detection and classification were accidentally the same
operation because we already knew the 5 colors to threshold for.

Pipeline (per detected hold):
    Contour -> filled mask -> sample HSV pixels inside the mask
    -> per-channel MEDIAN (robust to shadows/highlights/bolt holes)
    -> map (H, S, V) to a human-readable color name

Known simplification, not fixed here: hue is circular (0 wraps to
179), so a plain median can misbehave for colors that straddle that
boundary (red). We're not implementing circular statistics for V1;
just noting it as a documented limitation.
"""
import json

import cv2
import numpy as np

import hold_detection as hd

INPUT_PATH = "data/raw/crop_v1.png"
OUTPUT_IMAGE_PATH = "data/processed/classified_holds.png"
OUTPUT_JSON_PATH = "data/processed/classified_holds.json"

# Achromatic thresholds, checked before hue is consulted at all --
# at genuinely near-zero brightness or low saturation, hue is noisy/
# meaningless. BLACK_VALUE_MAX is deliberately very low: a DARK but
# SATURATED pixel (e.g. forest green, maroon) is still a color, not
# black -- we learned this the hard way when 50 caught several dark
# green/maroon holds before saturation was ever checked.
BLACK_VALUE_MAX = 20
WHITE_VALUE_MIN = 200
GRAY_SATURATION_MAX = 40

# (hue_min, hue_max, name) -- OpenCV hue range is 0-179.
HUE_RANGES = [
    (0, 10, "red"),
    (10, 20, "orange"),
    (20, 35, "yellow"),
    (35, 85, "green"),
    (85, 100, "cyan"),
    (100, 130, "blue"),
    (130, 150, "purple"),
    (150, 170, "pink"),
    (170, 180, "red"),
]

# Canonical swatch color per name, purely for the "expected" half of the
# visualization -- the "actual" half always shows the real sampled color.
NAME_TO_DISPLAY_BGR = {
    "red": (0, 0, 255), "orange": (0, 140, 255), "yellow": (0, 255, 255),
    "green": (0, 180, 0), "cyan": (255, 255, 0), "blue": (255, 0, 0),
    "purple": (200, 0, 150), "pink": (180, 105, 255),
    "black": (30, 30, 30), "gray": (160, 160, 160), "white": (255, 255, 255),
}


def classify_hsv(h, s, v):
    if v < BLACK_VALUE_MAX:
        return "black"
    if s < GRAY_SATURATION_MAX:
        return "white" if v > WHITE_VALUE_MIN else "gray"
    for lo, hi, name in HUE_RANGES:
        if lo <= h < hi:
            return name
    return "unknown"


def representative_color(image_bgr, image_hsv, contour):
    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)

    hsv_pixels = image_hsv[mask == 255]
    bgr_pixels = image_bgr[mask == 255]

    h, s, v = np.median(hsv_pixels, axis=0)
    b, g, r = np.median(bgr_pixels, axis=0)
    return (h, s, v), (int(b), int(g), int(r))


def build_hold_records(image_bgr, contours):
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    holds = []

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        (hue, sat, val), bgr = representative_color(image_bgr, image_hsv, contour)
        color_name = classify_hsv(hue, sat, val)

        holds.append({
            "id": i,
            "bbox": [x, y, w, h],
            "centroid": [cx, cy],
            "area": cv2.contourArea(contour),
            "color_name": color_name,
            "hsv": [round(hue, 1), round(sat, 1), round(val, 1)],
            "bgr": list(bgr),
        })

    return holds


def draw_classified(image_bgr, holds):
    output = image_bgr.copy()
    for hold in holds:
        x, y, w, h = hold["bbox"]
        sampled_bgr = tuple(hold["bgr"])
        name_bgr = NAME_TO_DISPLAY_BGR.get(hold["color_name"], (255, 255, 255))

        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)

        # Two small swatches above the box: left = actual sampled color,
        # right = canonical color for the assigned name. If they look
        # very different, that's a visible sign of a misclassification.
        swatch_y = max(y - 18, 0)
        cv2.rectangle(output, (x, swatch_y), (x + 15, swatch_y + 15), sampled_bgr, thickness=-1)
        cv2.rectangle(output, (x + 17, swatch_y), (x + 32, swatch_y + 15), name_bgr, thickness=-1)
        cv2.putText(output, hold["color_name"], (x + 36, swatch_y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    return output


def main():
    image = hd.load_image(INPUT_PATH)
    contours = hd.detect_holds(image)
    holds = build_hold_records(image, contours)

    result = draw_classified(image, holds)
    cv2.imwrite(OUTPUT_IMAGE_PATH, result)

    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(holds, f, indent=2)

    counts = {}
    for hold in holds:
        counts[hold["color_name"]] = counts.get(hold["color_name"], 0) + 1

    print(f"Classified {len(holds)} holds")
    for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<8} {count}")
    print(f"Visualization saved to {OUTPUT_IMAGE_PATH}")
    print(f"Structured data saved to {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
