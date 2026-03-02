#!/usr/bin/env python3
"""
Logo Clustering — Disjoint Set Union (Union-Find)

Groups company websites by visual similarity of their extracted logos
using perceptual hashing and Union-Find.

Pipeline:
   1 - Exact URL deduplication            O(n)
   2 - File content deduplication (MD5)   O(n)
   3 - Perceptual hash computation        O(n)
   4 - All-pairs similarity + DSU merge   O(n²)

Two merge strategies ( 4):
  Default — threshold-based two-tier gate:
    Tier 1:  pHash hamming distance ≤ T1  ->  union  (high confidence)
    Tier 2:  pHash ≤ T2  AND  dHash ≤ T2D ->  union  (cross-validated)
  --k N  — single-linkage agglomerative:
    Sort all pairs by pHash distance, merge until exactly N groups remain.
"""

import os
import sys
import json
import hashlib
import io
import shutil
import time
import subprocess
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image, ImageChops
import imagehash

#  try to import SVG renderer 
try:
    import cairosvg
    _SVG_BACKEND = "cairosvg"
except ImportError:
    _SVG_BACKEND = "rsvg"          # fallback to rsvg-convert CLI

# ══════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════
# --k N  ->  target number of clusters (single-linkage agglomerative)
_K_CLUSTERS    = None
for _i, _arg in enumerate(sys.argv):
    if _arg == "--k" and _i + 1 < len(sys.argv):
        _K_CLUSTERS = int(sys.argv[_i + 1])
        break

LOGO_DIR       = "extracted_logosp"
RESULTS_FILE   = "extraction_results.json"
OUTPUT_FILE    = "clustering_results.json"
CLUSTER_DIR    = "clusters"

HASH_SIZE      = 8                 # 8×8 = 64-bit hashes
RENDER_SIZE    = 256               # SVG -> PNG render target (px)
MIN_IMG_DIM    = 8                 # skip images smaller than this

# Alpha-compositing background (gray so white logos stay visible)
BG_COLOR       = (128, 128, 128)   # neutral gray
MIN_CONTRAST   = 20                # skip images with max channel range < this

# Hamming-distance thresholds (out of HASH_SIZE² = 64 bits)
T1_PHASH       = 10                 # tier-1: auto-union
T2_PHASH       = 12                # tier-2: needs dHash confirmation
T2_DHASH       = 10                # dHash ceiling for tier-2
NORM_SIZE      = 128               # normalize to square before hashing


# ══════════════════════════════════════════════════════════════════
#  DSU  (Disjoint Set Union  /  Union-Find)
# ══════════════════════════════════════════════════════════════════
class DSU:
    """Union-Find with path compression + union by rank."""
    __slots__ = ("_p", "_r")

    def __init__(self):
        self._p = {}   # parent
        self._r = {}   # rank

    #  core ops 
    def add(self, x):
        if x not in self._p:
            self._p[x] = x
            self._r[x] = 0

    def find(self, x):
        root = x
        while self._p[root] != root:
            root = self._p[root]
        # path compression (iterative)
        while self._p[x] != root:
            self._p[x], x = root, self._p[x]
        return root

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        # union by rank
        if self._r[a] < self._r[b]:
            a, b = b, a
        self._p[b] = a
        if self._r[a] == self._r[b]:
            self._r[a] += 1
        return True

    #  queries 
    def connected(self, a, b):
        return self.find(a) == self.find(b)

    def groups(self):
        g: dict[str, list[str]] = defaultdict(list)
        for x in self._p:
            g[self.find(x)].append(x)
        return dict(g)

    @property
    def n_groups(self):
        return len({self.find(x) for x in self._p})


# ══════════════════════════════════════════════════════════════════
#  Image loading  &  preprocessing
# ══════════════════════════════════════════════════════════════════

def _to_rgb(img):
    """Flatten any mode (RGBA, P, LA, L, …) to RGB on neutral gray bg.

    Gray background ensures white-on-transparent logos remain visible
    (white bg would make them hash identically to blank images).
    """
    bg_rgba = (*BG_COLOR, 255)
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, bg_rgba)
        bg.paste(img, mask=img.split()[3])
        return bg.convert("RGB")
    if img.mode == "P":
        if "transparency" in img.info:
            return _to_rgb(img.convert("RGBA"))
        return img.convert("RGB")
    if img.mode == "LA":
        bg = Image.new("LA", img.size, (BG_COLOR[0], 255))
        bg.paste(img, mask=img.split()[1])
        return bg.convert("RGB")
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _is_low_contrast(img):
    """Detect blank / nearly-uniform images (failed renders, invisible logos)."""
    extrema = img.getextrema()
    if isinstance(extrema[0], tuple):
        return max(mx - mn for mn, mx in extrema) < MIN_CONTRAST
    return (extrema[1] - extrema[0]) < MIN_CONTRAST


def _render_svg(path):
    """SVG -> RGB PIL Image  (cairosvg or rsvg-convert fallback)."""
    abspath = str(Path(path).resolve())

    if _SVG_BACKEND == "cairosvg":
        try:
            # Read SVG and fix common issues before rendering
            svg_bytes = Path(path).read_bytes()
            svg_text = svg_bytes.decode("utf-8", errors="replace")

            # Fix viewbox -> viewBox (SVG is case-sensitive, lowercase is invalid)
            svg_text = svg_text.replace("viewbox=", "viewBox=")

            # Add xmlns if missing
            if "xmlns" not in svg_text and "<svg" in svg_text:
                svg_text = svg_text.replace(
                    "<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1
                )

            # Remove style="display:block/none" which can hide content
            import re as _re
            svg_text = _re.sub(
                r'\s+style="display:\s*(?:block|none)[^"]*"', "", svg_text
            )

            # Detect all-white fills and invert to black
            _WHITE = frozenset(["#fff", "#ffffff", "white", "rgb(255,255,255)"])
            fills = _re.findall(r"""fill\s*=\s*['"]([^'"]+)['"]""", svg_text)
            if fills:
                non_none = [
                    f.lower().strip()
                    for f in fills
                    if f.lower().strip() not in ("none", "")
                ]
                if non_none and all(f in _WHITE for f in non_none):
                    for w in ("#fff", "#ffffff", "#FFF", "#FFFFFF", "white"):
                        svg_text = svg_text.replace(
                            'fill="' + w + '"', 'fill="#000000"'
                        )
                        svg_text = svg_text.replace(
                            "fill='" + w + "'", "fill='#000000'"
                        )

            png = cairosvg.svg2png(
                bytestring=svg_text.encode("utf-8"),
                output_width=RENDER_SIZE,
                output_height=RENDER_SIZE,
            )
            img = _to_rgb(Image.open(io.BytesIO(png)))
            if _is_low_contrast(img):
                return None
            return img
        except Exception:
            pass

    # fallback -> rsvg-convert (part of librsvg2)
    try:
        r = subprocess.run(
            ["rsvg-convert",
             "-w", str(RENDER_SIZE),
             "-h", str(RENDER_SIZE),

             abspath],
            capture_output=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout:
            return _to_rgb(Image.open(io.BytesIO(r.stdout)))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


def load_image(path):
    """Load any supported format -> RGB PIL Image."""
    if not os.path.exists(path):
        return None
    try:
        if path.lower().endswith(".svg"):
            return _render_svg(path)
        img = Image.open(path)
        img.load()
        if img.width < MIN_IMG_DIM or img.height < MIN_IMG_DIM:
            return None
        rgb = _to_rgb(img)
        if _is_low_contrast(rgb):
            return None      # blank / invisible logo
        return rgb
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════
#  Perceptual hashing  &  similarity
# ══════════════════════════════════════════════════════════════════

def _crop_to_content(img, threshold = 25):
    """Remove uniform-color borders so the hash focuses on actual logo content.

    Without this, small logos on large uniform backgrounds (e.g. SVGs rendered
    to 256x256 with gray bg) hash mostly as 'background', causing false unions.
    """
    w, h = img.size
    if w < 16 or h < 16:
        return img

    # Sample background from four corners
    corners = [
        img.getpixel((0, 0)),      img.getpixel((w - 1, 0)),
        img.getpixel((0, h - 1)),  img.getpixel((w - 1, h - 1)),
    ]
    bg_color = max(set(corners), key=corners.count)

    bg = Image.new(img.mode, img.size, bg_color)
    diff = ImageChops.difference(img, bg).convert("L")
    mask = diff.point(lambda x: 255 if x > threshold else 0)
    bbox = mask.getbbox()

    if bbox is None:
        return img  # entirely uniform

    # Add small padding around content
    pad = max(4, min(w, h) // 12)
    x1 = max(0, bbox[0] - pad)
    y1 = max(0, bbox[1] - pad)
    x2 = min(w, bbox[2] + pad)
    y2 = min(h, bbox[3] + pad)

    cropped = img.crop((x1, y1, x2, y2))
    if cropped.width >= 8 and cropped.height >= 8:
        return cropped
    return img


def compute_hashes(img):
    """Crop to content, normalize to square, then multi-hash."""
    img = _crop_to_content(img)
    img = img.resize((NORM_SIZE, NORM_SIZE), Image.LANCZOS)
    return {
        "p": imagehash.phash(img, hash_size=HASH_SIZE),
        "d": imagehash.dhash(img, hash_size=HASH_SIZE),
        "a": imagehash.average_hash(img, hash_size=HASH_SIZE),
    }


def similar(h1, h2):
    """Two-tier similarity gate.

    Tier 1:  pHash ≤ T1  ->  auto-union  (high confidence)
    Tier 2:  pHash ≤ T2 AND dHash ≤ T2D  ->  cross-validated union
    """
    pd = h1["p"] - h2["p"]
    if pd <= T1_PHASH:
        return True
    if pd <= T2_PHASH and (h1["d"] - h2["d"]) <= T2_DHASH:
        return True
    return False


# ══════════════════════════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    print("═" * 60)
    print("  LOGO CLUSTERING — Union-Find + Perceptual Hashing")
    print("═" * 60)

    #  load extraction results 
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    ok   = [r for r in results
            if r.get("downloaded_path") and not r.get("error")]
    fail = [r for r in results
            if r.get("error") or not r.get("downloaded_path")]

    print(f"\n  {len(ok)} logos OK · {len(fail)} failed · {len(results)} total")

    dsu = DSU()
    for r in ok:
        dsu.add(r["domain"])

    dom2file = {r["domain"]: r["downloaded_path"] for r in ok}

    # identical URL 
    print("\nGroup by identical logo URL")
    url_map: dict[str, list[str]] = defaultdict(list)
    for r in ok:
        url_map[r["logo_url"]].append(r["domain"])

    u1 = sum(dsu.union(ds[0], d)
             for ds in url_map.values() for d in ds[1:])
    print(f"  {u1} unions  ->  {dsu.n_groups} groups")

    # identical file content (MD5) 
    print("\nGroup by file content (MD5)")
    md5_map: dict[str, list[str]] = defaultdict(list)
    for dom, fp in dom2file.items():
        if os.path.exists(fp):
            h = hashlib.md5(Path(fp).read_bytes()).hexdigest()
            md5_map[h].append(dom)

    u2 = sum(dsu.union(ds[0], d)
             for ds in md5_map.values() for d in ds[1:])
    print(f"  {u2} unions  ->  {dsu.n_groups} groups")

    #  perceptual hashes 
    print("\n  Compute perceptual hashes")

    # pick one representative file per current DSU group
    reps = {}          # group_root -> filepath
    for root, members in dsu.groups().items():
        for m in members:
            fp = dom2file.get(m)
            if fp and os.path.exists(fp):
                reps[root] = fp
                break

    print(f"  {len(reps)} unique images to hash")

    hashes = {}
    n_fail = 0

    def _hash_one(item):
        root, fp = item
        img = load_image(fp)
        if img is None:
            return root, None
        return root, compute_hashes(img)

    workers = min(os.cpu_count() or 4, 8)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_hash_one, it) for it in reps.items()]
        for i, fut in enumerate(as_completed(futs), 1):
            root, h = fut.result()
            if h:
                hashes[root] = h
            else:
                n_fail += 1
            if i % 500 == 0:
                print(f"    {i}/{len(reps)} done …")

    print(f"  {len(hashes)} hashed · {n_fail} failed")

    #   4 — all-pairs comparison 
    print("\n  4 — Pairwise comparison + Union-Find")
    roots = list(hashes.keys())
    n = len(roots)
    total_pairs = n * (n - 1) // 2
    print(f"  {n} representatives  ->  {total_pairs:,} pairs to check")

    u4 = 0

    if _K_CLUSTERS is not None:
        #  k-based: single-linkage agglomerative via DSU 
        target_k = max(1, _K_CLUSTERS)
        print(f"  Mode: k = {target_k}  (single-linkage agglomerative)")

        # Collect all pairwise pHash distances
        edges = []
        for i in range(n):
            hi = hashes[roots[i]]
            for j in range(i + 1, n):
                d = hi["p"] - hashes[roots[j]]["p"]
                edges.append((d, i, j))
            if (i + 1) % 500 == 0:
                print(f"    distances: row {i+1}/{n}")

        # Sort ascending — merge most similar first
        edges.sort()
        print(f"  {len(edges):,} edges sorted")

        for dist, i, j in edges:
            if dsu.n_groups <= target_k:
                break
            ri, rj = roots[i], roots[j]
            if dsu.find(ri) != dsu.find(rj):
                dsu.union(ri, rj)
                u4 += 1

        print(f"  {u4} unions  ->  {dsu.n_groups} groups  "
              f"(target k={target_k})")
    else:
        #  threshold-based (original two-tier gate) 
        for i in range(n):
            ri = roots[i]
            hi = hashes[ri]
            for j in range(i + 1, n):
                rj = roots[j]
                # skip if already in the same component
                if dsu.find(ri) == dsu.find(rj):
                    continue
                if similar(hi, hashes[rj]):
                    dsu.union(ri, rj)
                    u4 += 1
            if (i + 1) % 500 == 0:
                print(f"    row {i+1}/{n}  ({u4} unions so far)")

        print(f"  {u4} unions  ->  {dsu.n_groups} groups")

    #  output 
    print("\n  Write results")

    groups = sorted(dsu.groups().values(), key=lambda g: (-len(g), g[0]))

    out = {
        "total_domains":  len(ok),
        "total_groups":   len(groups),
        "multi_groups":   sum(1 for g in groups if len(g) > 1),
        "singletons":     sum(1 for g in groups if len(g) == 1),
        "failed_domains": [r["domain"] for r in fail],
        "parameters": {
            "hash_size":   HASH_SIZE,
            "norm_size":   NORM_SIZE,
            "t1_phash":    T1_PHASH,
            "t2_phash":    T2_PHASH,
            "t2_dhash":    T2_DHASH,
            "k_clusters":  _K_CLUSTERS,
        },
        "time_s": round(time.time() - t0, 1),
        "groups": [],
    }

    for idx, members in enumerate(groups, 1):
        members.sort()
        rep = next((dom2file[m] for m in members if dom2file.get(m)), None)
        entry = {
            "group_id": idx,
            "count":    len(members),
            "domains":  members,
        }
        if rep:
            entry["representative_logo"] = rep
        out["groups"].append(entry)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    #  summary 
    elapsed = round(time.time() - t0, 1)
    print(f"\n{'═' * 60}")
    print(f"  ✓  {len(ok)} domains  ->  {len(groups)} groups   ({elapsed}s)")
    print(f"     multi-domain: {out['multi_groups']}      "
          f"singletons: {out['singletons']}")
    print(f"  ✓  saved to {OUTPUT_FILE}")
    print(f"{'═' * 60}\n")

    # top groups
    print("Top 25 largest groups:")
    for g in out["groups"][:25]:
        preview = ", ".join(g["domains"][:6])
        if g["count"] > 6:
            preview += f"  … +{g['count'] - 6}"
        print(f"  #{g['group_id']:>4d}  ({g['count']:>3d})  {preview}")

    #   6 — cluster folder 
    print(f"\n  6 — Copy logos to {CLUSTER_DIR}/ with cluster prefix")
    cluster_root = Path(CLUSTER_DIR)
    if cluster_root.exists():
        shutil.rmtree(cluster_root)
    cluster_root.mkdir(parents=True)

    copied = 0
    for entry in out["groups"]:
        gid = entry["group_id"]
        for dom in entry["domains"]:
            fp = dom2file.get(dom)
            if fp and os.path.exists(fp):
                basename = os.path.basename(fp)
                dst = cluster_root / f"{gid}_{basename}"
                shutil.copy2(fp, dst)
                copied += 1

    print(f"  {copied} images copied")
    print()


if __name__ == "__main__":
    main()
