"""
Stage 3: color-agnostic hold detection on a real photo, with full
visual debugging of every pipeline stage.

The actual detection logic lives in hold_detection.py (shared with
classify_hold_colors.py). This script's job is specifically to dump
every intermediate stage to disk so we can inspect WHERE the pipeline
succeeds or fails, rather than only judging the final boxed image.

Known, documented Version-1 limitations (see conversation history /
README for the full investigation that led to these numbers):
    - Holds whose saturation is genuinely close to the wall's (gray,
      white, brown) are missed -- not a bug, the chosen cue's blind spot.
    - A handful of real holds very close to the photo's edge are
      dropped by the border filter, traded off against correctly
      rejecting background objects (ceiling structure, wall trim).
    - Very small holds sit in a real precision/recall tension with
      MIN_CONTOUR_AREA: lowering it recovers some real holds but also
      reintroduces false positives (tape/number tags, wood grain).
"""
import os

import cv2

import hold_detection as hd

INPUT_PATH = "data/raw/crop_v1.png"
DEBUG_DIR = "data/processed/debug_generic"
OUTPUT_PATH = "data/processed/detected_holds_generic.png"

BOX_COLOR_BGR = (0, 0, 255)


def draw_detections(image_bgr, contours):
    output = image_bgr.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + w, y + h), BOX_COLOR_BGR, thickness=3)
    return output


def main():
    os.makedirs(DEBUG_DIR, exist_ok=True)

    image = hd.load_image(INPUT_PATH)

    saturation = hd.saturation_channel(image)
    cv2.imwrite(f"{DEBUG_DIR}/1_saturation_channel.png", saturation)

    otsu_value, _ = hd.otsu_threshold(saturation)
    _, raw_mask = cv2.threshold(saturation, hd.SATURATION_THRESHOLD, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f"{DEBUG_DIR}/2_raw_mask.png", raw_mask)

    cleaned = hd.clean_mask(raw_mask, hd.OPEN_KERNEL_SIZE, hd.CLOSE_KERNEL_SIZE)
    cv2.imwrite(f"{DEBUG_DIR}/3_cleaned_mask.png", cleaned)

    all_contours = hd.find_hold_contours(cleaned, hd.MIN_CONTOUR_AREA)
    contours = hd.drop_border_touching(all_contours, image.shape)
    result = draw_detections(image, contours)
    cv2.imwrite(OUTPUT_PATH, result)

    print(f"Otsu would have picked saturation threshold: {otsu_value:.1f} (not used)")
    print(f"Using manually tuned threshold: {hd.SATURATION_THRESHOLD}")
    print(f"{len(all_contours)} candidate blobs before border filtering")
    print(f"Dropped {len(all_contours) - len(contours)} blob(s) touching the image border")
    print(f"Kept {len(contours)} hold detections")
    print(f"Debug images saved to {DEBUG_DIR}/")
    print(f"Final result saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
