# Logo Similarity

## Overview

Given a Parquet file with **4,384 company domain entries** (3,416 unique domains
after deduplication), this pipeline extracts the primary logo from each website
and clusters them by visual similarity - producing groups of domains that share
the same brand identity, using perceptual hashing and Union-Find.

**Final results**: 3,351 logos extracted (98.1% success rate), organized into
**1,161 similarity groups** (359 multi-domain clusters + 802 singletons).

---

## Thinking Process & Decision Path

### The Core Question: How Do You Compare Logos at Scale?

**Perceptual hashing** is the natural fit for this problem. The challenge is fundamentally geometric: the same Mazda logo appearing on
`mazda-autohaus-abs-erlangen.de` and `mazda-autohaus-albers-doerpen.de` differs in resolution and format, not in structure. A 64-bit hash captures that structure, and Hamming distance gives a clean, tunable similarity metric (0 = identical, 64 = maximally different). It's fast, deterministic, interpretable, and requires no training data.

### Why Three Hash Types?

A single hash function has blind spots. I use three complementary ones:

- **pHash** (perceptual) - DCT-based frequency decomposition. Robust to
  scaling, compression artifacts, and minor color shifts. This is the primary comparison metric.
- **dHash** (difference) - captures horizontal gradient direction. Good at
  distinguishing layouts that pHash might conflate (e.g., a left-facing vs.
  right-facing arrow).
- **aHash** (average) - captures overall luminance distribution. Stored for
  completeness but not used in the merge gate (pHash + dHash proved sufficient).

All three are computed at `HASH_SIZE = 8` (8×8 = 64-bit hashes) after
normalizing images to 128×128 pixels.

### Why Union-Find (DSU)?

The clustering problem has a key property: **transitivity**. If logo A matches
logo B, and logo B matches logo C, then all three belong to the same brand -
even if A and C have a larger Hamming distance (e.g., A is an SVG, C is a
compressed JPEG of the same mark).

Union-Find (Disjoint Set Union) handles transitivity naturally through its
`union()` operation. It also:

- Requires **no pre-specified k** - groups emerge from the data
- Runs in **O(n² · α(n))** with path compression and union by rank, where
  α(n) is the inverse Ackermann function (effectively constant, ≤ 4 for any
  practical input size)
- Supports an optional `--k N` flag for single-linkage agglomerative merging
  when a fixed number of clusters is desired

### The Two-Tier Similarity Gate

A single pHash threshold is a trap. Set it too loose (≤ 14) and unrelated logos
merge. Set it too strict (≤ 6) and legitimate variants of the same brand stay
separated. The solution is a **two-tier gate**:

| Tier | Condition | Rationale |
|------|-----------|-----------|
| **Tier 1** | pHash distance ≤ 10 | High confidence - auto-union. These logos are structurally near-identical. |
| **Tier 2** | pHash ≤ 12 **AND** dHash ≤ 10 | Cross-validated - requires both hash types to agree before merging, filtering out false positives in the 10–12 range. |

The thresholds were tuned empirically by examining known brand clusters (AAMCO
with 221 regional sites, Mazda with 101, Culligan with 45+33) and verifying
zero false positives in the top 25 groups.

---

## Extraction Pipeline (`logo_extractor.py`)

### Architecture

The extractor processes all 3,416 unique domains asynchronously using `aiohttp`
with a concurrency limit of 40 simultaneous connections. For each domain, it
tries a cascade of strategies, scores all candidates, and downloads the
highest-scoring one.

### Strategy Cascade

Each domain goes through multiple extraction strategies. The first strategy to
find a valid candidate doesn't stop the search - all strategies run, and the
**highest-scoring** candidate across all strategies wins:

| # | Strategy | What it does | Logos found |
|---|----------|-------------|-------------|
| 1 | `<img>` tag analysis | Scans `<img>` tags in header/nav, scores by position, alt text, parent classes, URL keywords | 2,234 |
| 2 | `<link>` tag parsing | `<link rel="icon/apple-touch-icon/shortcut icon">` with size preference | 291 |
| 3 | Google Favicon API | `google.com/s2/favicons?domain=...&sz=128` - reliable last-resort | 209 |
| 4 | Playwright `<img>` | JS-rendered pages: same scoring heuristics evaluated in-browser | 192 |
| 5 | Inline SVG extraction | `<svg>` elements in header/nav with logo ancestors, serialized as data URIs | 166 |
| 6 | JSON-LD / Schema.org | Parses `<script type="application/ld+json">` for `logo` or `image` fields | 73 |
| 7 | Playwright inline SVG | Browser-rendered SVGs serialized with sanitization | 67 |
| 8 | Direct favicon | `/favicon.ico`, `/favicon.png`, `/apple-touch-icon.png` URLs | 47 |
| 9 | `<meta>` tags | `og:image`, `twitter:image`, `msapplication-TileImage` | 30 |
| 10 | Google Favicon (fallback) | Used when all HTML strategies fail and page couldn't be fetched | 14 |
| 11 | Playwright CSS background | Browser-evaluated `background-image: url(...)` on logo elements | 13 |
| 12 | CSS `background-image` | Logo classes with `background-image: url(...)` | 5 |
| 13 | DuckDuckGo icon API | `icons.duckduckgo.com/ip3/{domain}.ico` | 5 |
| 14 | WordPress REST API | `/wp-json/wp/v2/` for `site_logo` / `site_icon` (bypasses WAFs) | 3 |
| 15 | Playwright JSON-LD | JSON-LD parsed from JS-rendered pages | 2 |
|    | _No logo found_ | Domain unreachable, no valid candidates, WAF-blocked | 65 |

### Candidate Scoring System

Every candidate image gets a numerical score based on contextual signals.
The highest score wins for each domain. Key scoring factors:

**Positive signals:**
- In `<header>` element: **+30**, with logo-class ancestor combo: **+40** more
- In `<nav>` element: **+20**
- Grandparent is `<header>` or `<nav>`: **+25**
- URL/alt/class/id contains "logo" or "brand": **+80–100**
- Parent `<a>` links to homepage (`href="/"`): **+40**
- SVG format: **+20**, PNG: **+10**
- Early DOM position (first 3 images): **+20**

**Negative signals:**
- In `<footer>`: **−20**
- Dealer/franchise keywords (autohaus, händler, filiale): **−20**
- Partner/sponsor/certification keywords: **−25**
- Social media icons (facebook, twitter): **−40**
- Social media platform logos (instagram, youtube, linkedin, tiktok): **−60**
- Footer URL filename ("footer" in path): **−25**
- No logo/brand signal at all: **−40**
- Content directories (`/media/`, `/uploads/`): **−25**
- Third-party brand in alt text (domain-aware check): **−80**

#### Domain-Aware Brand Penalization

A subtle but critical edge case: `adecco.de` was extracting "BMW-Logo2.png"
from a partner showcase carousel. Both images had "logo" in their URL, but the
alt text ("BMW") didn't match the domain ("adecco"). The fix: if an image's
alt text names a brand that doesn't appear in the current domain, and "logo" is
in the URL path, apply a **−80 penalty**. This correctly ranks the Adecco logo
above partner logos on `adecco.de`, while leaving `bmw.de` unaffected.

### Body Image Filtering

By default, `<img>` tags are only considered if they're inside `<header>` or
`<nav>` elements. Images in the page body (hero banners, product photos, stock
images) are skipped. As a fallback, the first 5 DOM images with an explicit
"logo" or "brand" signal in their URL, class, alt, or ID are still accepted
regardless of position.

### SVG Handling

SVGs required special attention due to several rendering pitfalls:

1. **Lowercase `viewbox`** - Some sites (notably Airbnb) use `viewbox` instead
   of `viewBox`. CairoSVG silently produces a transparent 0×0 raster. The
   `_sanitize_svg()` function fixes the casing.

2. **Missing `xmlns`** - SVGs extracted from inline HTML lack the XML namespace.
   Added automatically: `xmlns="http://www.w3.org/2000/svg"`.

3. **`currentColor` fill** - Inline SVGs inherit CSS color, but standalone they
   render transparent. Replaced with `black`.

4. **`display: none`** - Some SVGs are hidden sprite sheets. Removed
   `display:none` and `display: none` from style attributes.

5. **All-white fills** - 94 logos (Honda, Baker Tilly, Vans, Renault) had
   `fill="#fff"` or `fill="white"` designed for dark backgrounds. When rendered
   standalone on white, they're invisible. Detected and inverted to black.

6. **Tiny SVGs (close buttons)** - 41 Baker Tilly domains returned 10×10
   `viewBox` close-button SVGs (391 bytes). The `_svg_too_small()` function
   rejects any SVG with viewBox dimensions under 20×20.

7. **Minimum bytes** - Inline SVG data URIs under 400 bytes are rejected
   (typically decorative icons or close buttons, not real logos).

### URL Validation

Before any candidate is scored, its URL is filtered:

- **Reject patterns**: tracking pixels (`pixel`, `spacer`, `blank`), avatars
  (`profile-photo`, `gravatar`), language flags (`/flags/`), third-party
  widgets (`cookiebot.com`, `onetrust.com`, `google-analytics.com`)
- **Domain check**: off-domain images with no "logo" signal in their alt
  text are rejected (prevents grabbing CDN banners)
- **Size check**: downloaded files under 500 bytes are discarded (1×1 tracking
  pixels, placeholder dots)

### Playwright Fallback

For JavaScript-heavy sites that serve empty HTML until client-side rendering,
the pipeline falls back to **Playwright** (headless Chromium). Domains qualify
for Playwright if the initial HTTP fetch returned no valid logo candidates.

Playwright sessions:
- Navigate with `domcontentloaded` wait + 2-second network idle
- Run the same scoring heuristics as the HTML parser, but in browser-evaluated
  JavaScript
- Serialize inline SVGs with the same sanitization rules
- Use `playwright-stealth` to bypass bot detection

**Threading note**: I initially parallelized Playwright with
`ThreadPoolExecutor(8)`, but Playwright's synchronous API is not thread-safe
(`Browser.new_context: Protocol error`). Reverted to sequential processing
with a 15-second timeout per domain.

### Captcha / WAF Detection

Sites behind Cloudflare, DataDome, or hCaptcha return challenge pages instead
of content. Detected by checking `<title>` for keywords like "captcha",
"security check", "just a moment", "attention required". These are routed
directly to API fallbacks (Clearbit → Google Favicon → DuckDuckGo → WP REST).

---

## Clustering Pipeline (`logo_clusterer.py`)

### Six-Phase Pipeline

```
Phase 1 → Group by identical logo URL          (O(n))
Phase 2 → Group by identical file content (MD5) (O(n))
Phase 3 → Compute perceptual hashes             (O(n))
Phase 4 → All-pairs comparison + DSU merge       (O(n²·α(n)))
Phase 5 → Write clustering_results.json
Phase 6 → Copy logos to clusters/ with group prefix
```

### Image Preprocessing

Before hashing, each image is normalized:

1. **SVG rendering** - SVGs are rendered to 256×256 PNG via CairoSVG (or
   `rsvg-convert` CLI fallback)
2. **Alpha compositing** - transparent images are composited onto **gray
   (#808080)** background. This prevents white-on-transparent logos from
   hashing as blank.
3. **Content cropping** - uniform borders are trimmed so logos on large
   canvases hash by their actual content
4. **Resize to 128×128** - Lanczos resampling to a fixed square, eliminating
   dimension-related hash differences

### Merge Logic

```python
if phash_distance <= 10:          # Tier 1: high confidence
    union(a, b)
elif phash_distance <= 12 and dhash_distance <= 10:  # Tier 2: cross-validated
    union(a, b)
```

### Optional Fixed-k Mode

`--k N` switches to single-linkage agglomerative clustering: all pairwise
pHash distances are sorted, and the closest pairs are merged (union) one by one
until exactly N groups remain.

```bash
python3 logo_clusterer.py --k 1000
```

---

## Input Data

The input `logos.snappy.parquet` contains **4,384 rows**, but only **3,416
unique domains** after deduplication ({4384 − 3416 = } 968 duplicate entries).
Deduplication is done via Python `set()` on the domain column before processing.

---

## Results

### Extraction

| Metric | Value |
|--------|-------|
| Unique domains processed | 3,416 |
| Logos found (URL extracted) | 3,351 (98.1%) |
| Files successfully downloaded | 3,318 (97.1%) |
| No logo / errors | 65 |

**File types on disk**: 1,628 PNG, 1,341 SVG, 177 JPG, 96 WebP, 76 ICO

### Clustering

| Metric | Value |
|--------|-------|
| Total similarity groups | 1,161 |
| Multi-domain clusters | 359 |
| Singleton groups | 802 |
| Avg. multi-domain cluster size | 7.0 |
| Largest cluster | 230 (AAMCO) |

**Group size distribution:**

| Size | Count |
|------|-------|
| 1 (singleton) | 802 |
| 2–5 | 214 |
| 6–10 | 110 |
| 11–25 | 22 |
| 26–50 | 8 |
| 51–100 | 4 |
| 100+ | 1 |

### Top 10 Largest Clusters

| # | Size | Brand | Example domains |
|---|------|-------|-----------------|
| 1 | 230 | AAMCO | aamco-bellevue.com, aamco-chesapeakeva.com, ... |
| 2 | 70 | Mazda | mazda-autohaus-abs-erlangen.de, mazda-autohaus-albers-doerpen.de, ... |
| 3 | 69 | Culligan | allthingswater.com, culliganadvantage.com, ... |
| 4 | 53 | Spitex | spitex-aarenord.ch, spitex-appenzellerland.ch, ... |
| 5 | 52 | Kia | kia-ahs-roehrnbach.de, kia-auto-center-weiterstadt.de, ... |
| 6 | 37 | Toyota | toyota.com.mt, toyotaakkoyunlu.com.tr, ... |
| 7 | 32 | Nestlé | nestle-caribbean.com, nestle-esar.com, ... |
| 8 | 31 | B. Braun | bbraun.bg, bbraun.ca, bbraun.cl, ... |
| 9 | 31 | Veolia | veolia.am, veolia.be, veolia.bg, ... |
| 10 | 29 | Airbnb | airbnb.am, airbnb.ba, airbnb.be, ... |

---

## Paths Not Taken

### Google Favicon as Primary Source
The `google_favicon_extractor.py` file is a standalone tool that downloads
favicons via Google's S2 API (`google.com/s2/favicons?domain=...&sz=128`).
It produces cleaner clustering (all regional sites → same favicon) but at the
cost of image quality (128×128 PNGs only, no SVGs). I kept it as a separate
utility but use the full multi-strategy extractor as the primary pipeline,
because it produces higher-fidelity logos (SVGs, full-size brand marks) that
better represent the actual brand identity.

### Playwright Parallelization
I attempted to parallelize Playwright with `ThreadPoolExecutor(8)` for speed.
It crashed immediately - Playwright's synchronous API is not thread-safe
(`Browser.new_context: Protocol error`). Switched to async Playwright was
considered but would require rewriting the sync integration layer. Sequential
with a 15-second timeout per domain was the pragmatic choice.

### Tighter Thresholds
I tested tighter thresholds (Tier 1 ≤ 8, Tier 2 ≤ 10) which produced more
groups but split legitimate brand clusters. The current thresholds
(T1=10, T2=12/10) were chosen because they produce zero false positives in
the top 25 groups while successfully merging regional variants.

---

## Limitations

- **Genuinely different brand images** - when the same brand uses completely
  different logos on different sites (e.g., dealer text logo vs. brand emblem),
  no perceptual hash can merge them without also merging unrelated logos.
  Culligan previously appeared as two clusters but now merges into one (69
  domains) thanks to improved extraction consistency.
- **WAF-blocked sites** - some domains return 403 on all requests (HTTP,
  Playwright, and API fallbacks). These account for most of the 65 failures.
- **JavaScript-only sites** - sites that require full JS execution to render
  any content depend on the Playwright fallback, which runs sequentially
  and adds extraction time.

---

## Project Structure

| File | Description |
|------|-------------|
| `logo_extractor.py` | Main extraction pipeline - multi-strategy extraction with scoring, SVG sanitization, Playwright fallback |
| `logo_clusterer.py` | Clustering pipeline - Union-Find with perceptual hashing (pHash + dHash), two-tier gate |
| `google_favicon_extractor.py` | Standalone Google Favicon API extractor (utility, not used in main pipeline) |
| `logos.snappy.parquet` | Input dataset (4,384 domain entries) |
| `requirements.txt` | Python dependencies |
| `extraction_results.json` | Per-domain extraction results (URL, strategy, path, scores) |
| `clustering_results.json` | Cluster groups, thresholds, and statistics |
| `extracted_logos/` | Downloaded logo files (3,318 files) |
| `clusters/` | Logos copied with `{group_id}_{filename}` prefix for inspection |

---

## How to Run

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium

# Step 1: Extract logos from all domains
python3 logo_extractor.py

# Step 2: Cluster by visual similarity
python3 logo_clusterer.py

# Optional: force exactly N groups
python3 logo_clusterer.py --k 1000
```

**Runtime**: extraction takes ~15 minutes (async HTTP + sequential Playwright
fallback), clustering takes ~30 seconds (all-pairs comparison of 3,300 hashes).

### Dependencies

```
aiohttp, beautifulsoup4, CairoSVG, ImageHash, lxml, numpy,
pandas, pillow, playwright, playwright-stealth, pyarrow
```
