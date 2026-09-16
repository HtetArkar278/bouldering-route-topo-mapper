"""
Stage 1 setup: generate a controlled synthetic "wall" image.

Why: we want a Version-1 test case where WE know the exact right answer
(how many holds, what color, where) before we test the pipeline against
messy real photos. If the algorithm gets this wrong, the bug is in our
code -- not in ambiguity about what "correct" even means.

Pipeline:
    Blank background -> place non-overlapping colored circles ("holds")
    -> draw them -> save image -> save ground truth (color/center/radius)

Deliberately NOT modeled yet (saved for Version 2 - realistic images):
    shadows, lighting gradients, wall texture, non-circular hold shapes,
    desaturated/ambiguous colors, occlusion, touching holds.
"""
import json
import random

import cv2
import numpy as np

# --- Configuration ---
IMAGE_PATH = "data/raw/synthetic_wall_v1.png"
GROUND_TRUTH_PATH = "data/raw/synthetic_wall_v1_ground_truth.json"

WIDTH, HEIGHT = 900, 1200
BACKGROUND_BGR = (190, 190, 185)  # flat light gray, like painted concrete

RANDOM_SEED = 42  # fixed seed -> reproducible layout every time we run this

MIN_RADIUS = 25
MAX_RADIUS = 45
MARGIN = 60          # keep holds away from image edges
MIN_GAP = 15         # minimum empty space between two holds' edges
MAX_PLACEMENT_ATTEMPTS = 500

# Pure, well-separated colors (BGR, since OpenCV loads/saves BGR).
# Chosen so their HSV hues are far apart -- easy to threshold individually.
HOLD_COLORS = {
    "red":    (0, 0, 255),
    "yellow": (0, 255, 255),
    "green":  (0, 255, 0),
    "blue":   (255, 0, 0),
    "purple": (255, 0, 255),
}
HOLDS_PER_COLOR = 3


def circles_overlap(c1, c2, min_gap):
    (x1, y1, r1) = c1
    (x2, y2, r2) = c2
    center_distance = np.hypot(x1 - x2, y1 - y2)
    return center_distance < (r1 + r2 + min_gap)


def place_holds(width, height, colors, holds_per_color, rng):
    """Randomly place non-overlapping circles, one list of dicts per hold."""
    placed = []  # list of (x, y, radius) for overlap checks
    holds = []

    for color_name, bgr in colors.items():
        for _ in range(holds_per_color):
            for _attempt in range(MAX_PLACEMENT_ATTEMPTS):
                radius = rng.randint(MIN_RADIUS, MAX_RADIUS)
                x = rng.randint(MARGIN + radius, width - MARGIN - radius)
                y = rng.randint(MARGIN + radius, height - MARGIN - radius)

                if not any(circles_overlap((x, y, radius), other, MIN_GAP) for other in placed):
                    placed.append((x, y, radius))
                    holds.append({
                        "color_name": color_name,
                        "bgr": list(bgr),
                        "center": [x, y],
                        "radius": radius,
                    })
                    break
            else:
                raise RuntimeError(
                    f"Could not place a non-overlapping '{color_name}' hold "
                    f"after {MAX_PLACEMENT_ATTEMPTS} attempts. "
                    "Try a larger canvas or fewer holds."
                )

    return holds


def draw_holds(image, holds):
    for hold in holds:
        x, y = hold["center"]
        cv2.circle(image, (x, y), hold["radius"], hold["bgr"], thickness=-1)
    return image


def main():
    rng = random.Random(RANDOM_SEED)

    image = np.full((HEIGHT, WIDTH, 3), BACKGROUND_BGR, dtype=np.uint8)
    holds = place_holds(WIDTH, HEIGHT, HOLD_COLORS, HOLDS_PER_COLOR, rng)
    image = draw_holds(image, holds)

    cv2.imwrite(IMAGE_PATH, image)
    with open(GROUND_TRUTH_PATH, "w") as f:
        json.dump(holds, f, indent=2)

    print(f"Placed {len(holds)} holds ({HOLDS_PER_COLOR} per color x {len(HOLD_COLORS)} colors)")
    print(f"Image saved to {IMAGE_PATH}")
    print(f"Ground truth saved to {GROUND_TRUTH_PATH}")


if __name__ == "__main__":
    main()
