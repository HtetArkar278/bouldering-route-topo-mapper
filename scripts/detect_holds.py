"""
Stage 1: detect climbing holds of several known colors and visualize them.

Pipeline:
    Load image -> BGR to HSV -> per-color threshold -> binary mask
    -> find contours -> filter by area -> record detection
    (color, bbox, centroid, area) -> draw -> save
    -> compare detected counts against ground truth

Baseline hypothesis this script encodes (important limitation, not a bug):
    "A hold's color can be found by thresholding a HSV range we already
    know in advance." This works here because WE chose 5 fixed colors
    when generating the synthetic wall. It will NOT work on a real photo
    with holds in unknown/arbitrary colors -- that's the problem
    Milestone 3 needs to solve (detect holds some other way, then
    classify color as a separate step).
"""
import json

import cv2
import numpy as np

IMAGE_PATH = "data/raw/synthetic_wall_v1.png"
GROUND_TRUTH_PATH = "data/raw/synthetic_wall_v1_ground_truth.json"
OUTPUT_PATH = "data/processed/detected_holds_v1.png"

MIN_CONTOUR_AREA = 200

# HSV ranges for each color we deliberately used when generating the
# synthetic wall. OpenCV hue runs 0-179 (half of the usual 0-359 degrees).
# Red wraps around the hue circle (179 -> 0), so it needs two ranges.
# S/V lower bounds are loose (100, not 255) to tolerate anti-aliased edge
# pixels, while still safely excluding the low-saturation gray background.
COLOR_HSV_RANGES = {
    "red":    [((0, 100, 100), (10, 255, 255)), ((170, 100, 100), (179, 255, 255))],
    "yellow": [((20, 100, 100), (40, 255, 255))],
    "green":  [((50, 100, 100), (70, 255, 255))],
    "blue":   [((110, 100, 100), (130, 255, 255))],
    "purple": [((140, 100, 100), (160, 255, 255))],
}

BOX_COLOR_BGR = (255, 255, 255)
TEXT_COLOR_BGR = (255, 255, 255)


def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {path}")
    return image


def build_mask(image_hsv, ranges):
    """OR together one or more HSV ranges into a single binary mask."""
    mask = np.zeros(image_hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(image_hsv, np.array(lower), np.array(upper))
    return mask


def detect_holds(image_bgr, color_ranges, min_area):
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    detections = []

    for color_name, ranges in color_ranges.items():
        mask = build_mask(image_hsv, ranges)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            moments = cv2.moments(contour)
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

            detections.append({
                "color_name": color_name,
                "bbox": [x, y, w, h],
                "centroid": [cx, cy],
                "area": area,
            })

    return detections


def draw_detections(image_bgr, detections):
    output = image_bgr.copy()
    for det in detections:
        x, y, w, h = det["bbox"]
        cv2.rectangle(output, (x, y), (x + w, y + h), BOX_COLOR_BGR, thickness=2)
        cv2.putText(
            output, det["color_name"], (x, max(y - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR_BGR, thickness=1,
            lineType=cv2.LINE_AA,
        )
    return output


def summarize_against_ground_truth(detections, ground_truth_path):
    with open(ground_truth_path) as f:
        ground_truth = json.load(f)

    expected_counts = {}
    for hold in ground_truth:
        expected_counts[hold["color_name"]] = expected_counts.get(hold["color_name"], 0) + 1

    detected_counts = {}
    for det in detections:
        detected_counts[det["color_name"]] = detected_counts.get(det["color_name"], 0) + 1

    all_colors = sorted(set(expected_counts) | set(detected_counts))
    print(f"{'color':<8} {'expected':>8} {'detected':>8} {'match':>6}")
    for color in all_colors:
        expected = expected_counts.get(color, 0)
        detected = detected_counts.get(color, 0)
        match = "yes" if expected == detected else "NO"
        print(f"{color:<8} {expected:>8} {detected:>8} {match:>6}")


def main():
    image = load_image(IMAGE_PATH)
    detections = detect_holds(image, COLOR_HSV_RANGES, MIN_CONTOUR_AREA)
    result = draw_detections(image, detections)

    cv2.imwrite(OUTPUT_PATH, result)
    print(f"Detected {len(detections)} holds total")
    print(f"Result saved to {OUTPUT_PATH}\n")

    summarize_against_ground_truth(detections, GROUND_TRUTH_PATH)


if __name__ == "__main__":
    main()
