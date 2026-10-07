"""
Stage 3a: color-agnostic hold detection on a real photo.

Hypothesis: holds (plastic, often glossy/matte-colored) are more
SATURATED than the gray/beige concrete wall, regardless of hue. If true,
thresholding the saturation channel alone -- no hardcoded colors -- should
find hold candidates.

Pipeline:
    Load image -> BGR to HSV -> isolate S channel -> Otsu threshold
    -> morphological opening (remove small noise) -> closing (fill gaps)
    -> find contours -> filter by area -> draw boxes -> save

Every intermediate stage is also saved to disk so we can visually inspect
WHERE the pipeline succeeds or fails, instead of only judging the final
result.

Known limitation we're deliberately testing, not hiding: holds with
low saturation (brown/gray/olive) may look too similar to the wall for
this cue alone to separate them. We'll inspect the debug images to see
whether that happens here.

UPDATE after inspecting results on a real photo:
Otsu auto-picked 84 here, which was too conservative -- it missed 9 real
(mostly small/dim) holds that a manually-tuned lower threshold of 45
recovers, at the cost of only a few new noise blobs (wood grain, a dark
shadow corner). We keep Otsu's value for comparison/logging but threshold
at the tuned value instead.

We also filter out any blob touching the image border. In this photo
that correctly removes background bleeding in from outside the wall
(ceiling structure, a wooden post edge) -- but it ALSO discards a few
real holds that happen to sit very close to the frame edge, since we
can't distinguish "grazes the edge by 2px" from "genuinely cut off by
the crop." That's an accepted, documented trade-off for Version 1, not
a bug: the practical fix is leaving margin around holds you care about
when framing a photo, not smarter code.
"""
import cv2
import numpy as np

INPUT_PATH = "data/raw/crop_v1.png"
DEBUG_DIR = "data/processed/debug_generic"
OUTPUT_PATH = "data/processed/detected_holds_generic.png"

MIN_CONTOUR_AREA = 400  # real photo holds are larger in pixel terms than the synthetic circles

OPEN_KERNEL_SIZE = 5    # removes small noise (drilled texture holes, chalk flecks)
CLOSE_KERNEL_SIZE = 9   # fills small gaps inside a hold's blob (shadows/highlights)

# Manually tuned after comparing against Otsu's auto-picked value (see module
# docstring). Otsu's 84 missed too many real holds on this photo.
SATURATION_THRESHOLD = 45

BOX_COLOR_BGR = (0, 0, 255)


def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {path}")
    return image


def saturation_channel(image_bgr):
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(image_hsv)
    return s


def otsu_threshold(saturation):
    threshold_value, mask = cv2.threshold(
        saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return threshold_value, mask


def clean_mask(mask, open_size, close_size):
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel)

    return closed


def find_hold_contours(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def drop_border_touching(contours, image_shape):
    """Discard blobs whose bounding box touches any image edge.

    A blob touching the frame border is either background bleeding in
    from outside the photographed wall, or a real hold truncated by the
    crop -- either way we can't fully characterize it, so we exclude it
    rather than report a partial/unreliable detection.
    """
    height, width = image_shape[:2]
    kept = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        touches_border = x == 0 or y == 0 or x + w == width or y + h == height
        if not touches_border:
            kept.append(contour)
    return kept


def draw_detections(image_bgr, contours):
    output = image_bgr.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + w, y + h), BOX_COLOR_BGR, thickness=3)
    return output


def main():
    import os
    os.makedirs(DEBUG_DIR, exist_ok=True)

    image = load_image(INPUT_PATH)

    saturation = saturation_channel(image)
    cv2.imwrite(f"{DEBUG_DIR}/1_saturation_channel.png", saturation)

    otsu_value, _ = otsu_threshold(saturation)
    _, raw_mask = cv2.threshold(saturation, SATURATION_THRESHOLD, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f"{DEBUG_DIR}/2_raw_mask.png", raw_mask)

    cleaned = clean_mask(raw_mask, OPEN_KERNEL_SIZE, CLOSE_KERNEL_SIZE)
    cv2.imwrite(f"{DEBUG_DIR}/3_cleaned_mask.png", cleaned)

    all_contours = find_hold_contours(cleaned, MIN_CONTOUR_AREA)
    contours = drop_border_touching(all_contours, image.shape)
    result = draw_detections(image, contours)
    cv2.imwrite(OUTPUT_PATH, result)

    print(f"Otsu would have picked saturation threshold: {otsu_value:.1f} (not used)")
    print(f"Using manually tuned threshold: {SATURATION_THRESHOLD}")
    print(f"{len(all_contours)} candidate blobs before border filtering")
    print(f"Dropped {len(all_contours) - len(contours)} blob(s) touching the image border")
    print(f"Kept {len(contours)} hold detections")
    print(f"Debug images saved to {DEBUG_DIR}/")
    print(f"Final result saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
