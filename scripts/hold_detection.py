"""
Shared color-agnostic hold detection pipeline.

Extracted out of detect_holds_generic.py once a second script
(classify_hold_colors.py) needed the exact same detection logic --
not similar code, identical code that has to stay in sync.

Pipeline:
    Load image -> BGR to HSV -> isolate S channel -> threshold
    -> morphological opening (remove small noise) -> closing (fill gaps)
    -> find contours -> filter by area -> drop border-touching blobs

See detect_holds_generic.py's history/comments for how these tuned
values were chosen (Otsu's auto-threshold was too conservative; a
manually tuned value recovered more real holds for little extra noise).
"""
import cv2

MIN_CONTOUR_AREA = 400
OPEN_KERNEL_SIZE = 5
CLOSE_KERNEL_SIZE = 9
SATURATION_THRESHOLD = 45


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


def detect_holds(image_bgr):
    """Run the full detection pipeline, returning a list of hold contours."""
    saturation = saturation_channel(image_bgr)
    _, mask = cv2.threshold(saturation, SATURATION_THRESHOLD, 255, cv2.THRESH_BINARY)
    cleaned = clean_mask(mask, OPEN_KERNEL_SIZE, CLOSE_KERNEL_SIZE)
    candidates = find_hold_contours(cleaned, MIN_CONTOUR_AREA)
    return drop_border_touching(candidates, image_bgr.shape)
