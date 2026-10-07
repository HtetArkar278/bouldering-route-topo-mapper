"""
Stage 5: group classified holds into candidate routes.

Baseline hypothesis -- explicitly a hypothesis to test, not an assumed
truth: "holds of the same color form one route." Testing this on a real
photo (see history/README) showed it CLEARLY FAILS: a pure color-only
MST zigzags across the entire wall, connecting holds no climber could
reach consecutively. Color alone does not predict route membership here
-- likely because this gym distinguishes routes with small numbered
tape tags, not hold color, and holds of one color get reused across
many unrelated routes.

This version adds a second, independent constraint: SPATIAL proximity.
We reuse the minimum spanning tree (MST) already built per color group,
and cut any edge longer than SPATIAL_LINK_THRESHOLD. The remaining
connected pieces become candidate sub-routes. This is single-linkage
clustering, implemented directly from the MST we already had -- no new
algorithm needed, no scikit-learn dependency required.

SPATIAL_LINK_THRESHOLD = 150px was chosen by inspecting the actual
distribution of MST edge lengths across all color groups on this photo:
there's a visible gap between 123px and 198px, the clearest break point
in the data. This is photo-specific (tied to camera distance), not a
universal constant -- a different photo would need re-tuning.

No ground truth exists for real routes on this wall, so results are
sanity-checked visually, not scored with precision/recall. That's a
realistic constraint: most real climbing photos won't come with labeled
routes either.
"""
import json

import cv2

import classify_hold_colors as chc

INPUT_IMAGE_PATH = "data/raw/crop_v1.png"
INPUT_HOLDS_PATH = "data/processed/classified_holds.json"
OUTPUT_IMAGE_PATH = "data/processed/route_groups.png"
OUTPUT_JSON_PATH = "data/processed/route_groups.json"

SPATIAL_LINK_THRESHOLD = 150


def group_by_color(holds):
    groups = {}
    for hold in holds:
        groups.setdefault(hold["color_name"], []).append(hold)
    return groups


def distance(p, q):
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


def minimum_spanning_tree(centroids):
    """Prim's algorithm. Small N per group (<20), so O(n^2) is plenty."""
    n = len(centroids)
    if n < 2:
        return []

    in_tree = [False] * n
    in_tree[0] = True
    nearest_dist = [distance(centroids[0], centroids[i]) for i in range(n)]
    nearest_parent = [0] * n
    edges = []

    for _ in range(n - 1):
        u = min((i for i in range(n) if not in_tree[i]), key=lambda i: nearest_dist[i])
        in_tree[u] = True
        edges.append((nearest_parent[u], u, nearest_dist[u]))

        for v in range(n):
            if not in_tree[v]:
                d = distance(centroids[u], centroids[v])
                if d < nearest_dist[v]:
                    nearest_dist[v] = d
                    nearest_parent[v] = u

    return edges


def build_color_group(color_name, holds):
    centroids = [tuple(h["centroid"]) for h in holds]
    edges = minimum_spanning_tree(centroids)
    return {
        "color_name": color_name,
        "hold_ids": [h["id"] for h in holds],
        "centroids": centroids,
        "mst_edges": edges,
    }


def assign_spatial_clusters(group, threshold):
    """Union-find over the color group's MST, keeping only short edges.

    Connected components after this cut are spatially-coherent
    sub-clusters within one color -- single-linkage clustering using
    the MST we already have, instead of a new algorithm.
    """
    n = len(group["centroids"])
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j, d in group["mst_edges"]:
        if d <= threshold:
            union(i, j)

    cluster_ids, next_id = [], {}
    for idx in range(n):
        root = find(idx)
        if root not in next_id:
            next_id[root] = len(next_id)
        cluster_ids.append(next_id[root])

    return cluster_ids


def split_into_routes(group, threshold):
    cluster_ids = assign_spatial_clusters(group, threshold)
    n_clusters = len(set(cluster_ids))

    members_by_cluster = {}
    for idx, cid in enumerate(cluster_ids):
        members_by_cluster.setdefault(cid, []).append(idx)

    routes = []
    for cid, idxs in sorted(members_by_cluster.items()):
        sub_centroids = [group["centroids"][i] for i in idxs]
        sub_hold_ids = [group["hold_ids"][i] for i in idxs]
        sub_edges = minimum_spanning_tree(sub_centroids)
        name = f"{group['color_name']}_{cid + 1}" if n_clusters > 1 else group["color_name"]

        routes.append({
            "route_name": name,
            "color_name": group["color_name"],
            "hold_ids": sub_hold_ids,
            "centroids": sub_centroids,
            "mst_edges": [[i, j, round(d, 1)] for i, j, d in sub_edges],
        })

    return routes


def draw_routes(image_bgr, routes):
    output = image_bgr.copy()
    for route in routes:
        color_bgr = chc.NAME_TO_DISPLAY_BGR.get(route["color_name"], (255, 255, 255))
        centroids = route["centroids"]

        if len(centroids) == 1:
            cx, cy = centroids[0]
            cv2.circle(output, (cx, cy), 5, color_bgr, thickness=-1)
            cv2.circle(output, (cx, cy), 9, (120, 120, 120), thickness=1)
            continue

        for i, j, _ in route["mst_edges"]:
            cv2.line(output, centroids[i], centroids[j], color_bgr, thickness=2, lineType=cv2.LINE_AA)
        for cx, cy in centroids:
            cv2.circle(output, (cx, cy), 6, color_bgr, thickness=-1)
            cv2.circle(output, (cx, cy), 6, (255, 255, 255), thickness=1)

        cx, cy = centroids[0]
        cv2.putText(output, route["route_name"], (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
        cv2.putText(output, route["route_name"], (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, thickness=1, lineType=cv2.LINE_AA)

    return output


def main():
    image = cv2.imread(INPUT_IMAGE_PATH)
    with open(INPUT_HOLDS_PATH) as f:
        holds = json.load(f)

    groups = [build_color_group(color, members) for color, members in group_by_color(holds).items()]

    routes = []
    for group in groups:
        routes.extend(split_into_routes(group, SPATIAL_LINK_THRESHOLD))

    result = draw_routes(image, routes)
    cv2.imwrite(OUTPUT_IMAGE_PATH, result)

    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(routes, f, indent=2)

    multi_hold = [r for r in routes if len(r["hold_ids"]) > 1]
    strays = [r for r in routes if len(r["hold_ids"]) == 1]

    print(f"{len(holds)} holds -> {len(routes)} groups after color + spatial splitting")
    print(f"  {len(multi_hold)} candidate routes with 2+ holds")
    print(f"  {len(strays)} stray single-hold groups (not really \"routes\")\n")
    for route in sorted(multi_hold, key=lambda r: -len(r["hold_ids"])):
        print(f"  {route['route_name']:<10} {len(route['hold_ids'])} holds  ids={route['hold_ids']}")

    print(f"\nVisualization saved to {OUTPUT_IMAGE_PATH}")
    print(f"Structured data saved to {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
