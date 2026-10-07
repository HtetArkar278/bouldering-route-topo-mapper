"""
Manual point annotation tool: click to mark hold centers on an image.

Why: when we disagree with the algorithm about what's a hold, the most
useful feedback isn't "it looks wrong" -- it's exact pixel coordinates
we can directly compare against the algorithm's detected centroids.
This also doubles as the ground-truth annotation tool we'll need later
(Version 5) to measure precision/recall on real photos, where -- unlike
the synthetic wall -- we don't get ground truth for free.

Controls:
    Left click   - mark a point
    Right click  - remove the nearest marked point (undo a misclick)
    s            - save annotations and quit
    q / Esc      - quit WITHOUT saving

Usage:
    python scripts/annotate_points.py <image_path> <output_json_path>

Output:
    <output_json_path>            - list of [x, y] in ORIGINAL image
                                     pixel coordinates (not the scaled
                                     display coordinates)
    <output_json_path with
     .json replaced by _preview.png> - the image with your clicks drawn
                                        on it, for a visual sanity check
"""
import argparse
import json
import os

import cv2

DISPLAY_MAX_HEIGHT = 900  # keep the window on-screen on typical laptop displays
POINT_RADIUS = 6
POINT_COLOR_BGR = (0, 255, 255)
REMOVE_RADIUS_PX = 15  # how close a right-click must be (in display pixels) to delete a point


def parse_args():
    parser = argparse.ArgumentParser(description="Click to annotate hold centers on an image.")
    parser.add_argument("image_path", help="Path to the image to annotate")
    parser.add_argument("output_json_path", help="Where to save clicked points (original-image pixel coords)")
    return parser.parse_args()


def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {path}")
    return image


def make_display_image(image, max_height):
    height, width = image.shape[:2]
    scale = min(1.0, max_height / height)
    display = cv2.resize(image, (int(width * scale), int(height * scale)))
    return display, scale


def redraw(base_display, points):
    canvas = base_display.copy()
    for i, (x, y) in enumerate(points):
        cv2.circle(canvas, (x, y), POINT_RADIUS, POINT_COLOR_BGR, thickness=-1)
        cv2.putText(canvas, str(i), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, POINT_COLOR_BGR, thickness=1, lineType=cv2.LINE_AA)
    cv2.putText(canvas, f"{len(points)} points  |  left-click add, right-click remove, s=save, q=quit",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(canvas, f"{len(points)} points  |  left-click add, right-click remove, s=save, q=quit",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    return canvas


def nearest_point_index(points, x, y, max_dist):
    best_idx, best_dist = None, max_dist
    for i, (px, py) in enumerate(points):
        dist = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
        if dist < best_dist:
            best_idx, best_dist = i, dist
    return best_idx


def main():
    args = parse_args()
    image = load_image(args.image_path)
    display_base, scale = make_display_image(image, DISPLAY_MAX_HEIGHT)

    points = []  # in DISPLAY coordinates while annotating

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            idx = nearest_point_index(points, x, y, REMOVE_RADIUS_PX)
            if idx is not None:
                points.pop(idx)

    window_name = "Annotate holds"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    print("Left-click each hold center. Right-click to remove a misclick.")
    print("Press 's' to save and quit, 'q' or Esc to quit without saving.")

    while True:
        cv2.imshow(window_name, redraw(display_base, points))
        key = cv2.waitKey(16) & 0xFF
        if key == ord("s"):
            break
        if key == ord("q") or key == 27:
            points = None
            break

    cv2.destroyAllWindows()

    if points is None:
        print("Quit without saving.")
        return

    original_coords = [[round(x / scale), round(y / scale)] for x, y in points]
    with open(args.output_json_path, "w") as f:
        json.dump(original_coords, f, indent=2)

    preview_path = args.output_json_path.replace(".json", "_preview.png")
    cv2.imwrite(preview_path, redraw(display_base, points))

    print(f"Saved {len(original_coords)} points to {args.output_json_path}")
    print(f"Preview saved to {preview_path}")


if __name__ == "__main__":
    main()
