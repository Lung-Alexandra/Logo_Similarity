"""
Logo Extractor 
====================================
Extracts logos from a list of company domains using multiple strategies:

1. HTML parsing - <link rel="icon">, <meta og:image>, <img> with logo heuristics
2. Structured data - JSON-LD, schema.org logo references
3. Manifest.json - PWA manifests with icon definitions
4. Google Favicon API - reliable fallback for favicons
5. Clearbit Logo API - high-quality company logos (free tier)
6. WordPress REST API - site_logo / site_icon from /wp-json/ (for WAF-blocked WP sites)

The extractor uses async HTTP to process thousands of domains efficiently,
with retry logic, proper timeouts, and graceful error handling.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import warnings

import aiohttp
import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONCURRENCY = 40            # max parallel requests
TIMEOUT_SECONDS = 15        # per-request timeout
MAX_RETRIES = 2             # retries on transient failures
OUTPUT_DIR = Path("extracted_logos")
RESULTS_FILE = Path("extraction_results.json")
RESULTS_CSV = Path("extraction_results.csv")
LOG_FILE = Path("extraction.log")
DOWNLOAD_IMAGES = True      # whether to download actual image files
MIN_IMAGE_SIZE = 200        # minimum bytes for a valid logo image
MIN_LOGO_SIZE = 500         # minimum bytes to consider it a real logo (not just a dot)
USE_PLAYWRIGHT = True       # enable Playwright fallback for dynamic sites
PW_CONCURRENCY = 8          # max parallel Playwright browser tabs

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
]

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("logo_extractor")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LogoResult:
    """Holds the extraction result for one domain."""
    domain: str
    logo_url: Optional[str] = None
    source_strategy: Optional[str] = None  # which strategy found it
    downloaded_path: Optional[str] = None
    error: Optional[str] = None
    http_status: Optional[int] = None
    extraction_time_ms: int = 0
    _best_score: int = 0  # internal: best candidate score from HTML pass


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def get_user_agent(domain):
    """Deterministic but varied user-agent per domain."""
    idx = hash(domain) % len(USER_AGENTS)
    return USER_AGENTS[idx]


def normalize_url(url, base_url):
    """Resolve a potentially relative URL against a base."""
    if not url or not url.strip():
        return None
    url = url.strip()
    # data URIs - skip (except inline SVG data URIs which we generate)
    if url.startswith("data:image/svg+xml"):
        return url
    if url.startswith("data:"):
        return None
    # protocol-relative
    if url.startswith("//"):
        url = "https:" + url
    # relative
    if not url.startswith(("http://", "https://")):
        url = urljoin(base_url, url)
    return url


def is_likely_logo_url(url):
    """Heuristic: does this URL look like a logo resource?"""
    if not url:
        return False
    lower = url.lower()
    logo_keywords = ["logo", "brand", "header-img", "site-icon", "company"]
    return any(kw in lower for kw in logo_keywords)


def is_valid_image_url(url):
    """Check that URL points to an image-like resource."""
    if not url:
        return False
    lower = url.lower()
    
    # Extract just the filename/last path segment for tighter matching
    path_end = lower.rsplit("/", 1)[-1] if "/" in lower else lower
    
    # reject tracking pixels, spacers, common non-logo patterns
    # Only match on the filename part to avoid false positives from directory names
    reject_filename = ["spacer", "pixel", "tracking", "1x1", "blank.",
                       "spinner", "loading", "placeholder"]
    if any(r in path_end for r in reject_filename):
        return False
    
    # These keywords in the full URL are usually non-logo (more specific patterns)
    reject_full = ["profile-photo", "profile-pic", "profile-image",
                   "avatar/", "avatar.", "user-avatar",
                   "/wallpaper", "bg-pattern", "bg-texture",
                   "/flags/"]
    if any(r in lower for r in reject_full):
        return False
    
    # reject third-party logos (cookie banners, analytics, captchas, etc.)
    third_party_reject = ["cookielaw.org", "cookiebot.com", "onetrust.com",
                          "cookie-cdn", "cookie-script", "trustarc",
                          "evidon.com", "quantcast", "termly.io",
                          "powered_by_logo", "gdpr", "consent-manager",
                          "captcha-delivery.com", "/captcha/", "datadome.co",
                          "hcaptcha.com", "recaptcha"]
    if any(r in lower for r in third_party_reject):
        return False
    return True


def score_logo_candidate(url, tag_context):
    """
    Assign a priority score to a logo candidate.
    Higher = better candidate.
    Improved heuristics to prefer the main site/brand logo over
    dealer logos, partner logos, footer logos, etc.
    """
    score = 0
    lower_url = url.lower()
    url_path = urlparse(url).path.lower()

    # ---- URL-based signals ----
    has_logo_signal = False  # will be set True if any logo/brand keyword found
    if "logo" in lower_url:
        score += 40
        has_logo_signal = True

    # Penalize third-party brand logos shown as partner/client showcases
    # e.g. BMW-Logo2, Amazon-Logo on an Adecco page
    # Only penalize if the alt text brand does NOT match the current domain
    _alt_text = tag_context.get("alt", "").strip().lower()
    _parent_href = tag_context.get("parent_href", "").lower()
    _domain = tag_context.get("domain", "").lower()
    if _alt_text and _alt_text not in ("logo", "home", "") and len(_alt_text) < 30:
        # Check if the alt text brand appears in the domain
        _alt_clean = re.sub(r'[^a-z0-9]', '', _alt_text)
        _dom_clean = re.sub(r'[^a-z0-9]', '', _domain)
        if _alt_clean and _dom_clean and _alt_clean not in _dom_clean:
            # Alt text names something NOT in the domain - likely a partner logo
            # But only penalize if "logo" is in the URL (partner logo carousel)
            if "logo" in lower_url:
                score -= 80
    # If parent link goes to a career/jobs page for another company, penalize
    if _parent_href and any(kw in _parent_href for kw in
                            ["/karriere-", "/career-", "/partner/", "/client/"]):
        score -= 60

    # SVG logos are typically the primary brand logo
    if lower_url.endswith(".svg") or "/svg" in lower_url:
        score += 20
    elif lower_url.endswith(".png"):
        score += 10
    elif lower_url.endswith(".jpg") or lower_url.endswith(".jpeg"):
        score += 5

    # ---- Tag context signals ----
    alt = tag_context.get("alt", "").lower()
    cls = tag_context.get("class", "").lower()
    id_ = tag_context.get("id", "").lower()
    parent_cls = tag_context.get("parent_class", "").lower()
    parent_id = tag_context.get("parent_id", "").lower()
    grandparent_cls = tag_context.get("grandparent_class", "").lower()
    grandparent_id = tag_context.get("grandparent_id", "").lower()
    grandparent_tag = tag_context.get("grandparent_tag", "").lower()
    is_in_header = tag_context.get("is_in_header", False)
    is_in_footer = tag_context.get("is_in_footer", False)
    is_in_nav = tag_context.get("is_in_nav", False)
    dom_position = tag_context.get("dom_position", 999)

    all_context = f"{alt} {cls} {id_} {parent_cls} {parent_id} {grandparent_cls} {grandparent_id} {url_path}"

    # ---- NAVIGATION / TOP-BAR LOGO (highest priority) ----
    # These are almost always THE brand logo
    nav_logo_patterns = ["nav_logo", "nav-logo", "navbar-logo", "navbar_logo",
                         "eut_nav_logo", "main-logo", "main_logo", "site-logo",
                         "site_logo", "top-logo", "top_logo",
                         "site-header__logo", "header__logo", "header-logo",
                         "header_logo", "masthead-logo", "masthead__logo",
                         "branding__logo", "branding-logo",
                         "navigation-logo", "navigation_logo",
                         "home-logolink", "home-logo", "home_logo",
                         "logolink"]
    for pattern in nav_logo_patterns:
        if pattern in cls or pattern in id_ or pattern in parent_cls or pattern in parent_id:
            score += 100
            has_logo_signal = True
            break
    # Also check grandparent for BEM-style nesting
    else:
        for pattern in nav_logo_patterns:
            if pattern in grandparent_cls or pattern in grandparent_id:
                score += 80
                has_logo_signal = True
                break

    # parent_id or parent_class exactly "logo" is extremely strong signal
    if parent_id == "logo" or parent_cls == "logo":
        score += 80
        has_logo_signal = True
    elif "logo" in parent_id or "logo" in parent_cls:
        has_logo_signal = True
        # But not if it's a "dealer" logo or secondary logo
        if "dealer" not in parent_cls and "dealer" not in parent_id:
            score += 50
        else:
            score += 20  # dealer header logos get less priority

    # Grandparent with logo
    if "logo" in grandparent_cls or "logo" in grandparent_id:
        has_logo_signal = True
        if "dealer" not in grandparent_cls and "dealer" not in grandparent_id:
            score += 40
        else:
            score += 15

    # Image's own class/id with "logo"
    if "logo" in cls or "logo" in id_:
        has_logo_signal = True
        # Boost nav-related logo classes much more
        if "nav" in cls or "nav" in id_:
            score += 70
        elif "dealer" in cls or "dealer" in id_:
            score += 15
        else:
            score += 50

    # Alt text containing logo
    if "logo" in alt:
        score += 30
        has_logo_signal = True

    # Brand keywords in class/id
    for text in [cls, id_, parent_cls, parent_id, alt]:
        if "brand" in text:
            score += 25
            has_logo_signal = True
        if "site-logo" in text or "site_logo" in text:
            score += 60
            has_logo_signal = True
        if "header-logo" in text or "header_logo" in text:
            if "dealer" not in text:
                score += 55
            else:
                score += 10
            has_logo_signal = True
        if "navbar-brand" in text:
            score += 50
            has_logo_signal = True
        if "custom-logo" in text:
            score += 50
            has_logo_signal = True

    # ---- Position bonuses ----
    if is_in_header:
        score += 30
        # COMBO: in header AND parent/grandparent has "logo" = very strong signal
        if "logo" in parent_cls or "logo" in parent_id or "logo" in grandparent_cls:
            score += 40  # extra combo bonus
    if is_in_nav:
        score += 20
    if grandparent_tag in ("header", "nav"):
        score += 25

    # Earlier in DOM = more likely the main logo
    if dom_position <= 3:
        score += 20
    elif dom_position <= 10:
        score += 10

    # ---- Penalties ----
    if is_in_footer:
        score -= 20

    # Dealer / franchise logo - these are secondary to the brand logo
    dealer_keywords = ["dealer", "franchise", "branch", "filiale",
                       "autohaus", "haendler", "händler", "standort"]
    for kw in dealer_keywords:
        if kw in all_context:
            score -= 20
            break

    # Partner/sponsor logos
    partner_keywords = ["partner", "sponsor", "client", "customer",
                        "affiliate", "certification", "award", "badge",
                        "trust", "seal", "payment", "association"]
    for kw in partner_keywords:
        if kw in all_context:
            score -= 25
            break

    # URL path hints at non-primary logo
    if any(x in url_path for x in ["/logos/", "/partner", "/client",
                                     "/sponsor", "/certification", "/award"]):
        score -= 20

    # "capital" in URL (like award/label logos)
    if "capital" in lower_url and "logo" in lower_url:
        score -= 15

    if "icon" in lower_url and "logo" not in lower_url:
        score -= 10
    if "social" in lower_url or "facebook" in lower_url or "twitter" in lower_url:
        score -= 40
    if "flag" in lower_url:
        score -= 20
    if "favicon" in lower_url:
        score -= 15

    # ---- Homepage-link bonus ----
    # Images inside <a href="/"> are almost always the main site logo
    parent_href = tag_context.get("parent_href", "")
    if parent_href in ("/", "", "./", "#"):
        if parent_href == "/" or parent_href == "./":
            score += 40  # strong signal: links to homepage
            has_logo_signal = True
    elif parent_href:
        # Full-URL homepage links like https://example.com/
        from urllib.parse import urlparse as _urlparse
        _ph = _urlparse(parent_href)
        if _ph.scheme in ('http', 'https') and _ph.path.rstrip('/') == '':
            score += 40
            has_logo_signal = True

    # ---- Penalty: NO logo/brand signal at all ----
    # Images that only score from position (header, DOM order) without any
    # logo/brand keyword are likely banners, hero images, or stock photos.
    if not has_logo_signal:
        score -= 40

    # ---- Penalty: content image paths ----
    # Paths like /media/, /uploads/, /files/, /styles/, /public/ are
    # content-managed images, rarely logos.
    content_dirs = ["/media/", "/uploads/", "/files/", "/styles/",
                    "/public/", "/active_storage/", "/wp-content/uploads/",
                    "/sites/", "/push_menu/", "/promo/"]
    if any(d in url_path for d in content_dirs):
        # Only penalize if no explicit logo signal in the path itself
        if "logo" not in url_path:
            score -= 25

    return score


def file_extension_from_url(url):
    """Guess file extension from URL."""
    parsed = urlparse(url)
    path = parsed.path.lower()
    for ext in [".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp"]:
        if path.endswith(ext):
            return ext
    # Check for format in query params
    if "format=png" in url.lower():
        return ".png"
    if "format=svg" in url.lower():
        return ".svg"
    return ".png"  # default


def sanitize_filename(domain):
    """Create a safe filename from a domain."""
    return re.sub(r"[^\w\-.]", "_", domain)


_WHITE_FILLS = frozenset(["#fff", "#ffffff", "white", "rgb(255,255,255)"])

_MIN_SVG_VIEWBOX = 20  # reject SVGs whose viewBox is smaller than 20x20


def _svg_too_small(svg_text):
    """Return True if the SVG viewBox/width/height is tiny (icon, not logo)."""
    m = re.search(r'viewBox\s*=\s*["\']\s*[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)', svg_text, re.I)
    if m:
        w, h = float(m.group(1)), float(m.group(2))
        if w < _MIN_SVG_VIEWBOX and h < _MIN_SVG_VIEWBOX:
            return True
    # Also check explicit width/height attributes on root <svg>
    root = re.search(r'<svg[^>]*>', svg_text, re.I)
    if root:
        tag = root.group(0)
        wm = re.search(r'\bwidth\s*=\s*["\']([\d.]+)', tag)
        hm = re.search(r'\bheight\s*=\s*["\']([\d.]+)', tag)
        if wm and hm:
            w, h = float(wm.group(1)), float(hm.group(1))
            if w < _MIN_SVG_VIEWBOX and h < _MIN_SVG_VIEWBOX:
                return True
    return False


def _sanitize_svg(data):
    """Fix common SVG issues so the file renders correctly standalone.

    - viewbox (lowercase) -> viewBox (SVG spec is case-sensitive)
    - Add xmlns if missing
    - Replace currentcolor with black
    - Remove style display:none/block that hides content
    - Invert all-white fills to black (invisible on white background)
    """
    if isinstance(data, bytes):
        text = data.decode("utf-8", errors="replace")
    else:
        text = data
    # viewbox -> viewBox  (SVG spec requires camelCase)
    text = re.sub(r'\bviewbox\s*=', 'viewBox=', text)
    # Add xmlns if missing
    if "xmlns" not in text and "<svg" in text:
        text = text.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
    # currentcolor -> black
    text = re.sub(r"""fill\s*=\s*['"]currentcolor['"]""", 'fill="#000000"', text, flags=re.I)
    text = re.sub(r'fill\s*:\s*currentcolor', 'fill:#000000', text, flags=re.I)
    text = re.sub(r"""stroke\s*=\s*['"]currentcolor['"]""", 'stroke="#000000"', text, flags=re.I)
    text = re.sub(r'stroke\s*:\s*currentcolor', 'stroke:#000000', text, flags=re.I)
    # Remove display:none / display:block that hides content
    text = re.sub(r'style\s*=\s*"[^"]*display\s*:\s*(none|block)[^"]*"', '', text)
    # Detect all-white fills and invert to black so logo is visible
    fills = re.findall(r"""fill\s*=\s*['"]([^'"]+)['"]""", text)
    if fills:
        non_none = [f.lower().strip() for f in fills if f.lower().strip() not in ("none", "")]
        if non_none and all(f in _WHITE_FILLS for f in non_none):
            for w in ("#fff", "#ffffff", "#FFF", "#FFFFFF", "white"):
                text = text.replace('fill="' + w + '"', 'fill="#000000"')
                text = text.replace("fill='" + w + "'", "fill='#000000'")
    return text.encode("utf-8")


# ---------------------------------------------------------------------------
# Extraction strategies (from HTML)
# ---------------------------------------------------------------------------

def extract_from_link_tags(soup, base_url):
    """
    Strategy 1: <link> tags - rel="icon", rel="apple-touch-icon",
    rel="shortcut icon", and also rel="image_src".
    """
    candidates = []
    link_selectors = [
        {"rel": re.compile(r"icon", re.I)},
    ]

    for link in soup.find_all("link"):
        rel = link.get("rel", [])
        if isinstance(rel, list):
            rel_str = " ".join(rel).lower()
        else:
            rel_str = str(rel).lower()

        href = link.get("href")
        if not href:
            continue

        url = normalize_url(href, base_url)
        if not url or not is_valid_image_url(url):
            continue

        # apple-touch-icon is usually high-quality
        if "apple-touch-icon" in rel_str:
            candidates.append((url, 50))
        elif "icon" in rel_str and url.lower().endswith(".svg"):
            # SVG favicons are high-quality vector logos
            candidates.append((url, 65))
        elif "icon" in rel_str:
            # Prefer larger icons
            sizes = link.get("sizes", "")
            size_score = 0
            if sizes:
                try:
                    w = int(sizes.split("x")[0])
                    size_score = min(w // 10, 30)  # up to 30 bonus
                except (ValueError, IndexError):
                    pass
            candidates.append((url, 40 + size_score))
        elif "image_src" in rel_str:
            candidates.append((url, 35))

    return candidates


def extract_from_meta_tags(soup, base_url):
    """
    Strategy 2: <meta> tags - og:image, og:logo, twitter:image,
    msapplication-TileImage, etc.
    """
    candidates = []

    meta_properties = {
        "og:image": 45,
        "og:logo": 70,
        "twitter:image": 40,
        "msapplication-tileimage": 35,
        "image": 30,
    }

    for meta in soup.find_all("meta"):
        prop = (meta.get("property") or meta.get("name") or "").lower()
        content = meta.get("content", "")
        if not content:
            continue

        for key, score in meta_properties.items():
            if prop == key:
                url = normalize_url(content, base_url)
                if url and is_valid_image_url(url):
                    # Boost if URL contains "logo"
                    bonus = 20 if is_likely_logo_url(url) else 0
                    candidates.append((url, score + bonus))

    return candidates


def extract_from_img_tags(soup, base_url, domain=""):
    """
    Strategy 3: <img> tags with logo-related class/id/alt/src attributes.
    Enhanced with grandparent context and DOM position awareness.
    """
    candidates = []

    # Pre-detect if there's a <header> or <nav> element
    header_el = soup.find("header")
    footer_el = soup.find("footer")
    nav_el = soup.find("nav")

    for idx, img in enumerate(soup.find_all("img")):
        src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
        # Fallback to srcset if no src - take the first URL from srcset
        if not src:
            srcset = img.get("srcset", "")
            if srcset:
                first_entry = srcset.split(",")[0].strip().split()[0]
                if first_entry:
                    src = first_entry
        if not src:
            continue

        url = normalize_url(src, base_url)
        if not url or not is_valid_image_url(url):
            continue

        # Gather context - including grandparent and position
        parent = img.parent
        grandparent = parent.parent if parent else None

        # Check if img is inside <header>, <footer>, <nav>
        is_in_header = False
        is_in_footer = False
        is_in_nav = False
        for ancestor in img.parents:
            tag_name = getattr(ancestor, "name", "")
            acls = ' '.join(ancestor.get('class', [])).lower() if hasattr(ancestor, 'get') else ''
            arole = (ancestor.get('role', '') or '').lower() if hasattr(ancestor, 'get') else ''
            if tag_name == "header" or arole == 'banner':
                is_in_header = True
            if tag_name == "footer":
                is_in_footer = True
            if tag_name == "nav" or arole == 'navigation' or 'navigation' in acls or 'navbar' in acls:
                is_in_nav = True

        # ---- SKIP images outside header/nav ----
        # Logo-ul principal e mereu în header sau nav, nu în body.
        # Fallback: acceptăm primele 5 img din DOM dacă au "logo" explicit
        if not is_in_header and not is_in_nav:
            # Allow early-DOM images that have explicit logo signal in URL/class/alt/id
            _img_cls = ' '.join(img.get('class', [])).lower()
            _img_id = (img.get('id', '') or '').lower()
            _img_alt = (img.get('alt', '') or '').lower()
            _p_cls = ' '.join(parent.get('class', [])).lower() if parent and hasattr(parent, 'get') else ''
            _p_id = (parent.get('id', '') or '').lower() if parent and hasattr(parent, 'get') else ''
            has_explicit_logo = ('logo' in url.lower() or 'logo' in _img_cls or
                                 'logo' in _img_id or 'logo' in _img_alt or
                                 'logo' in _p_cls or 'logo' in _p_id or
                                 'brand' in _img_cls or 'brand' in _p_cls)
            if not (idx < 5 and has_explicit_logo):
                continue  # skip body images

        # Check if img is inside an <a> linking to the homepage
        parent_a = img.find_parent("a")
        parent_href = ""
        if parent_a:
            parent_href = parent_a.get("href", "").strip()

        context = {
            "alt": img.get("alt", ""),
            "class": " ".join(img.get("class", [])),
            "id": img.get("id", ""),
            "parent_class": " ".join(parent.get("class", [])) if parent and hasattr(parent, "get") else "",
            "parent_id": parent.get("id", "") if parent and hasattr(parent, "get") else "",
            "grandparent_class": " ".join(grandparent.get("class", [])) if grandparent and hasattr(grandparent, "get") else "",
            "grandparent_id": grandparent.get("id", "") if grandparent and hasattr(grandparent, "get") else "",
            "grandparent_tag": grandparent.name if grandparent else "",
            "is_in_header": is_in_header,
            "is_in_footer": is_in_footer,
            "is_in_nav": is_in_nav,
            "dom_position": idx,
            "parent_href": parent_href,
            "domain": domain,
        }

        score = score_logo_candidate(url, context)

        # Only keep candidates with positive score (some logo signal)
        if score > 0:
            candidates.append((url, score))

    return candidates


def extract_from_svg_inline(soup, base_url):
    """
    Strategy 4: Inline <svg> inside anchors/headers that link to homepage,
    or inside elements with 'logo' in class/id within header/nav.
    For inline SVGs without img fallback, serialize as data URI.
    """
    import base64
    candidates = []

    def _fix_currentcolor(svg_str):
        """Replace fill/stroke='currentcolor' with black so SVG renders standalone."""
        svg_str = re.sub(r'fill\s*=\s*["\']currentcolor["\']', 'fill="#000000"', svg_str, flags=re.I)
        svg_str = re.sub(r'fill\s*:\s*currentcolor', 'fill:#000000', svg_str, flags=re.I)
        svg_str = re.sub(r'stroke\s*=\s*["\']currentcolor["\']', 'stroke="#000000"', svg_str, flags=re.I)
        svg_str = re.sub(r'stroke\s*:\s*currentcolor', 'stroke:#000000', svg_str, flags=re.I)
        return svg_str

    # Approach 1: SVG inside <a> tags linking to homepage with <img> fallback
    for a_tag in soup.find_all("a", href=True):
        href = a_tag.get("href", "").strip()
        # Only consider links to homepage-like paths
        is_homepage_link = href in ("/", "#", "", "./")
        if not is_homepage_link:
            # Check for full-URL homepage links (e.g. https://example.com/)
            parsed_href = urlparse(href)
            if parsed_href.scheme in ('http', 'https') and parsed_href.path.rstrip('/') == '':
                is_homepage_link = True
            # Also match domain-relative root
            elif href.rstrip('/') == '' or href == base_url.rstrip('/'):
                is_homepage_link = True
        if not is_homepage_link:
            continue

        svg = a_tag.find("svg")
        if not svg:
            continue

        # Check for nearby <img> fallback
        img = a_tag.find("img")
        if img:
            src = img.get("src") or img.get("data-src")
            if src:
                url = normalize_url(src, base_url)
                if url and is_valid_image_url(url):
                    candidates.append((url, 75))  # high confidence
        else:
            # No <img> fallback - serialize the inline SVG as data URI
            svg_str = _fix_currentcolor(str(svg))
            if len(svg_str) > 200 and not _svg_too_small(svg_str):
                encoded = base64.b64encode(svg_str.encode('utf-8')).decode('ascii')
                data_url = f"data:image/svg+xml;base64,{encoded}"
                # Boost score if anchor has brand/logo class or aria-label
                a_cls = ' '.join(a_tag.get('class', [])).lower()
                a_aria = (a_tag.get('aria-label', '') or '').lower()
                score = 80
                if 'brand' in a_cls or 'logo' in a_cls:
                    score = 90
                if 'home' in a_aria:
                    score += 5
                candidates.append((data_url, score))

    # Approach 2: Inline SVGs in logo-class containers (header/nav)
    # These are often THE brand logo (e.g. Kia, Tesla, modern SPA sites)
    for svg in soup.find_all("svg"):
        # Skip SVGs that are clearly UI icons (not logos)
        svg_cls = ' '.join(svg.get('class', [])).lower()
        aria_hidden = svg.get('aria-hidden', '').lower()
        if aria_hidden == 'true':
            continue  # decorative/UI icon
        icon_cls_hints = ['icon', 'toggle', 'menu', 'chevron', 'arrow',
                          'spinner', 'loading', 'caret', 'close']
        if any(h in svg_cls for h in icon_cls_hints):
            continue

        has_logo_ancestor = False
        in_header = False
        in_nav = False
        in_footer = False
        # Only check up to 4 levels of ancestors (avoid distant false positives)
        for i, ancestor in enumerate(svg.parents):
            if i > 4:
                break
            if not hasattr(ancestor, 'get'):
                continue
            acls = ' '.join(ancestor.get('class', [])).lower()
            aid = (ancestor.get('id', '') or '').lower()
            tag_name = getattr(ancestor, 'name', '')
            arole = (ancestor.get('role', '') or '').lower()
            if 'logo' in acls or 'logo' in aid or 'brand' in acls or 'brand' in aid:
                has_logo_ancestor = True
            if tag_name == 'header' or 'header' in acls or arole == 'banner':
                in_header = True
            if tag_name == 'nav' or 'nav-' in acls or 'nav ' in (acls + ' ') or arole == 'navigation':
                in_nav = True
            if tag_name == 'footer' or 'footer' in acls:
                in_footer = True

        if has_logo_ancestor and (in_header or in_nav) and not in_footer:
            svg_str = _fix_currentcolor(str(svg))
            # Must be non-trivial (small SVGs are just icons/arrows)
            if len(svg_str) > 200 and not _svg_too_small(svg_str):
                encoded = base64.b64encode(svg_str.encode('utf-8')).decode('ascii')
                data_url = f"data:image/svg+xml;base64,{encoded}"
                score = 85
                if in_header and in_nav:
                    score = 95
                candidates.append((data_url, score))

    return candidates


def extract_from_json_ld(soup, base_url):
    """
    Strategy 5: Structured data - JSON-LD with schema.org logo property.
    """
    candidates = []

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue

        # Handle arrays of JSON-LD objects
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            logo = item.get("logo")
            if isinstance(logo, str):
                url = normalize_url(logo, base_url)
                if url:
                    # Penalize dealer/franchise logos in JSON-LD
                    jld_score = 90
                    lower = url.lower()
                    dealer_hints = ["dealer-logo", "dealer_logo", "/dealer/",
                                    "dealer-logos", "dealer_logos",
                                    "haendler", "franchise", "filiale"]
                    if any(kw in lower for kw in dealer_hints):
                        jld_score = 50
                    candidates.append((url, jld_score))
            elif isinstance(logo, dict):
                logo_url = logo.get("url") or logo.get("contentUrl")
                if logo_url:
                    url = normalize_url(logo_url, base_url)
                    if url:
                        jld_score = 90
                        lower = url.lower()
                        dealer_hints = ["dealer-logo", "dealer_logo", "/dealer/",
                                        "dealer-logos", "dealer_logos",
                                        "haendler", "franchise", "filiale"]
                        if any(kw in lower for kw in dealer_hints):
                            jld_score = 50
                        candidates.append((url, jld_score))
            # Also check image property of Organization
            image = item.get("image")
            if isinstance(image, str) and is_likely_logo_url(image):
                url = normalize_url(image, base_url)
                if url:
                    candidates.append((url, 70))
            elif isinstance(image, dict):
                img_url = image.get("url") or image.get("contentUrl")
                if img_url and is_likely_logo_url(img_url):
                    url = normalize_url(img_url, base_url)
                    if url:
                        candidates.append((url, 70))

    return candidates


def extract_from_css_background(soup, base_url):
    """
    Strategy 6: Elements with inline style background-image that contain 'logo'.
    """
    candidates = []
    bg_re = re.compile(r"url\(['\"]?([^)'\"]+(logo|brand)[^)'\"]*)[\'\"]?\)", re.I)

    for tag in soup.find_all(style=True):
        style = tag.get("style", "")
        match = bg_re.search(style)
        if match:
            url = normalize_url(match.group(1), base_url)
            if url and is_valid_image_url(url):
                candidates.append((url, 55))

    return candidates


# ---------------------------------------------------------------------------
# Manifest.json extraction
# ---------------------------------------------------------------------------

async def extract_from_manifest(session,soup,base_url):
    """
    Strategy 7: Fetch manifest.json for PWA icons.
    """
    candidates = []
    manifest_link = soup.find("link", rel="manifest")
    if not manifest_link:
        return candidates

    manifest_href = manifest_link.get("href")
    if not manifest_href:
        return candidates

    manifest_url = normalize_url(manifest_href, base_url)
    if not manifest_url:
        return candidates

    try:
        async with session.get(
            manifest_url,
            timeout=aiohttp.ClientTimeout(total=8),
            ssl=False,
        ) as resp:
            if resp.status == 200:
                data = await resp.json(content_type=None)
                icons = data.get("icons", [])
                for icon in icons:
                    src = icon.get("src")
                    if src:
                        url = normalize_url(src, manifest_url)
                        if url:
                            # Skip framework default icons
                            framework_defaults = ["jhipster", "angular_logo",
                                                  "react-logo", "vue-logo",
                                                  "next-logo", "nuxt-icon",
                                                  "vite.", "webpack-logo"]
                            if any(kw in url.lower() for kw in framework_defaults):
                                continue
                            sizes = icon.get("sizes", "")
                            size_score = 0
                            if sizes and "x" in sizes:
                                try:
                                    w = int(sizes.split("x")[0])
                                    size_score = min(w // 10, 30)
                                except ValueError:
                                    pass
                            candidates.append((url, 35 + size_score))
    except Exception:
        pass

    return candidates


# ---------------------------------------------------------------------------
# Fallback strategies (API-based, no HTML needed)
# ---------------------------------------------------------------------------

async def try_google_favicon(session, domain):
    """Google S2 Favicon service - reliable fallback."""
    url = f"https://www.google.com/s2/favicons?domain={domain}&sz=128"
    try:
        async with session.get(
            url, timeout=aiohttp.ClientTimeout(total=8), ssl=False
        ) as resp:
            if resp.status == 200:
                data = await resp.read()
                if len(data) > MIN_IMAGE_SIZE:
                    return url
    except Exception:
        pass
    return None


async def try_clearbit_logo(session, domain):
    """Clearbit Logo API - high-quality logos, free."""
    url = f"https://logo.clearbit.com/{domain}"
    try:
        async with session.get(
            url, timeout=aiohttp.ClientTimeout(total=8),
            ssl=False, allow_redirects=True,
        ) as resp:
            if resp.status == 200:
                data = await resp.read()
                if len(data) > MIN_IMAGE_SIZE:
                    return url
    except Exception:
        pass
    return None


async def try_duckduckgo_icon(session, domain):
    """DuckDuckGo icon service fallback."""
    url = f"https://icons.duckduckgo.com/ip3/{domain}.ico"
    try:
        async with session.get(
            url, timeout=aiohttp.ClientTimeout(total=8), ssl=False,
        ) as resp:
            if resp.status == 200:
                data = await resp.read()
                if len(data) > MIN_IMAGE_SIZE:
                    return url
    except Exception:
        pass
    return None


async def _curl_json(url, timeout = 10):
    """Fetch JSON via curl subprocess - bypasses TLS fingerprint-based WAFs."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "curl", "-s", "-m", str(timeout),
            "-H", "Accept: application/json",
            "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout + 5)
        if proc.returncode == 0 and stdout:
            return json.loads(stdout)
    except Exception:
        pass
    return None


async def try_wp_json_logo(session, domain):
    """
    WordPress REST API - extract site_logo and site_icon from /wp-json/.
    Works even when the main page returns 403 (WAF/geo-block) because
    the REST API endpoint is often left unblocked.
    Uses curl subprocess when aiohttp is blocked by TLS-fingerprint WAFs.
    Returns list of (url, score, strategy_name) tuples.
    """
    results: list[tuple[str, int, str]] = []
    wp_json_url = f"https://{domain}/wp-json/"

    data = None
    # Try aiohttp first (faster, no subprocess)
    try:
        async with session.get(
            wp_json_url,
            timeout=aiohttp.ClientTimeout(total=10),
            ssl=False,
            allow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
            },
        ) as resp:
            if resp.status == 200:
                ct = resp.headers.get("Content-Type", "")
                if "json" in ct or "javascript" in ct:
                    data = await resp.json(content_type=None)
    except Exception:
        pass

    # Fallback to curl if aiohttp was blocked (WAF/TLS fingerprint)
    if not isinstance(data, dict):
        data = await _curl_json(wp_json_url)

    if not isinstance(data, dict):
        return results

    site_icon_url = data.get("site_icon_url", "")
    site_logo_id = data.get("site_logo", 0)

    # Fetch the site_logo media object (higher quality than icon)
    if site_logo_id and isinstance(site_logo_id, int) and site_logo_id > 0:
        media_url = f"https://{domain}/wp-json/wp/v2/media/{site_logo_id}"
        mdata = None
        try:
            async with session.get(
                media_url,
                timeout=aiohttp.ClientTimeout(total=8),
                ssl=False,
                headers={"Accept": "application/json"},
            ) as mresp:
                if mresp.status == 200:
                    mdata = await mresp.json(content_type=None)
        except Exception:
            pass
        if not isinstance(mdata, dict):
            mdata = await _curl_json(media_url)

        if isinstance(mdata, dict):
            logo_src = mdata.get("source_url", "")
            if logo_src:
                results.append((logo_src, 90, "wp_json_logo"))

    # site_icon_url is directly available in the root response
    if site_icon_url and isinstance(site_icon_url, str) and site_icon_url.startswith("http"):
        results.append((site_icon_url, 60, "wp_json_icon"))

    return results


# ---------------------------------------------------------------------------
# Image downloader
# ---------------------------------------------------------------------------

async def download_google_favicon_fallback(session,domain, output_dir):
    """Last-resort download: grab the Google S2 favicon with minimal validation.

    Uses relaxed checks (only MIN_IMAGE_SIZE, no content-type rejection) so
    that domains whose logos failed stricter validation can still be rescued.
    Returns the saved file path or None.
    """
    url = f"https://www.google.com/s2/favicons?domain={domain}&sz=128"
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=10),
            ssl=False,
        ) as resp:
            if resp.status != 200:
                return None
            data = await resp.read()
            if len(data) < MIN_IMAGE_SIZE:
                return None

            ct = resp.headers.get("Content-Type", "").lower()
            if "png" in ct:
                ext = ".png"
            elif "ico" in ct or "x-icon" in ct:
                ext = ".ico"
            elif "svg" in ct:
                ext = ".svg"
            elif "jpeg" in ct or "jpg" in ct:
                ext = ".jpg"
            else:
                ext = ".png"  # Google usually returns PNG

            filename = sanitize_filename(domain) + ext
            filepath = output_dir / filename
            filepath.write_bytes(data)
            return str(filepath)
    except Exception:
        return None


async def download_logo(session,logo_url, domain, output_dir):
    """Download the logo and save to disk. Returns the saved file path."""

    # Handle inline SVG data URIs (from extract_from_svg_inline)
    if logo_url.startswith("data:image/svg+xml;base64,"):
        import base64 as b64
        try:
            encoded = logo_url.split(",", 1)[1]
            svg_data = b64.b64decode(encoded)
            if len(svg_data) < 400:
                return None  # too small, probably not a real logo
            svg_text = svg_data.decode('utf-8', errors='replace')
            if _svg_too_small(svg_text):
                return None  # tiny icon (close button, chevron, etc.)
            svg_data = _sanitize_svg(svg_data)
            filename = sanitize_filename(domain) + ".svg"
            filepath = output_dir / filename
            filepath.write_bytes(svg_data)
            return str(filepath)
        except Exception:
            return None

    ext = file_extension_from_url(logo_url)
    filename = sanitize_filename(domain) + ext
    filepath = output_dir / filename

    aiohttp_ok = False
    try:
        async with session.get(
            logo_url,
            timeout=aiohttp.ClientTimeout(total=15),
            ssl=False,
        ) as resp:
            if resp.status == 200:
                data = await resp.read()
                ct = resp.headers.get("Content-Type", "").lower()

                # Validate the response
                valid = True
                if len(data) < MIN_IMAGE_SIZE:
                    valid = False
                elif "text/html" in ct or "text/plain" in ct:
                    valid = False
                elif len(data) > 250_000 and "svg" not in ct:
                    valid = False
                elif ext != ".ico" and len(data) < MIN_LOGO_SIZE:
                    valid = False
                # Reject tiny SVG icons (close buttons, arrows, etc.)
                elif (ext == ".svg" or "svg" in ct) and _svg_too_small(data.decode('utf-8', errors='replace')):
                    valid = False

                if valid:
                    # Sanitize SVG content
                    if ext == ".svg" or "svg" in ct:
                        data = _sanitize_svg(data)
                    # Update extension based on actual content type
                    if "svg" in ct:
                        ext = ".svg"
                    elif "png" in ct:
                        ext = ".png"
                    elif "jpeg" in ct or "jpg" in ct:
                        ext = ".jpg"
                    elif "gif" in ct:
                        ext = ".gif"
                    elif "webp" in ct:
                        ext = ".webp"
                    elif "x-icon" in ct or "ico" in ct:
                        ext = ".ico"

                    filename = sanitize_filename(domain) + ext
                    filepath = output_dir / filename
                    filepath.write_bytes(data)
                    aiohttp_ok = True
    except Exception:
        pass

    if aiohttp_ok:
        return str(filepath)

    # Fallback: use curl for downloads blocked by TLS-fingerprint WAFs
    if logo_url.startswith("http"):
        try:
            proc = await asyncio.create_subprocess_exec(
                "curl", "-s", "-m", "12", "-L",
                "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "-o", str(filepath), "-w", "%{http_code} %{size_download} %{content_type}",
                logo_url,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=18)
            if proc.returncode == 0 and stdout:
                parts = stdout.decode().strip().split(None, 2)
                if len(parts) >= 2:
                    http_code, size = parts[0], int(parts[1])
                    ct_curl = parts[2] if len(parts) > 2 else ""
                    if http_code == "200" and size >= MIN_IMAGE_SIZE:
                        if "text/html" not in ct_curl and size <= 250_000:
                            # Update extension based on content type
                            for ct_hint, ct_ext in [("svg", ".svg"), ("png", ".png"),
                                                     ("jpeg", ".jpg"), ("gif", ".gif"),
                                                     ("webp", ".webp")]:
                                if ct_hint in ct_curl:
                                    new_fp = output_dir / (sanitize_filename(domain) + ct_ext)
                                    if new_fp != filepath:
                                        filepath.rename(new_fp)
                                        filepath = new_fp
                                    break
                            return str(filepath)
            # Clean up failed download
            if filepath.exists():
                filepath.unlink()
        except Exception:
            if filepath.exists():
                filepath.unlink()

    return None


# ---------------------------------------------------------------------------
# Main extraction pipeline for a single domain
# ---------------------------------------------------------------------------

async def extract_logo_for_domain(session,domain,output_dir,semaphore):
    """
    Full extraction pipeline for one domain.
    Tries multiple strategies in priority order.
    """
    result = LogoResult(domain=domain)
    start = time.monotonic()

    async with semaphore:
        base_url = f"https://{domain}"
        html = None

        # ------- Step 1: Fetch the homepage HTML -------
        # Try with and without www. prefix
        domain_variants = [domain]
        if not domain.startswith("www."):
            domain_variants.append(f"www.{domain}")

        for attempt in range(MAX_RETRIES + 1):
            for d_variant in domain_variants:
                for scheme in ["https", "http"]:
                    url = f"{scheme}://{d_variant}"
                    try:
                        async with session.get(
                            url,
                            timeout=aiohttp.ClientTimeout(total=TIMEOUT_SECONDS),
                            ssl=False,
                            allow_redirects=True,
                            headers={
                                "User-Agent": get_user_agent(domain),
                                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                                "Accept-Language": "en-US,en;q=0.9",
                                "Accept-Encoding": "gzip, deflate",
                                "Connection": "keep-alive",
                            },
                        ) as resp:
                            result.http_status = resp.status
                            if resp.status == 200:
                                html = await resp.text(errors="replace")
                                base_url = str(resp.url)  # follow redirects
                                break
                    except (aiohttp.ClientError, asyncio.TimeoutError, UnicodeDecodeError):
                        continue
                    except Exception:
                        continue
                if html:
                    break
            if html:
                break
            if attempt < MAX_RETRIES:
                await asyncio.sleep(1)

        # ------- Step 2: Parse HTML and apply strategies -------
        all_candidates: list[tuple[str, int, str]] = []  # (url, score, strategy)

        if html:
            try:
                soup = BeautifulSoup(html, "lxml")
            except Exception:
                soup = BeautifulSoup(html, "html.parser")

            # Detect captcha/challenge pages (DataDome, Cloudflare, etc.)
            title_tag = soup.find("title")
            page_title = title_tag.string.lower().strip() if title_tag and title_tag.string else ""
            captcha_signals = ["captcha", "security check", "challenge",
                               "just a moment", "attention required",
                               "ddos-guard", "access denied", "bot verification"]
            is_captcha_page = any(s in page_title for s in captcha_signals)
            if not is_captcha_page:
                # Also check for captcha-delivery.com in the body
                body_str = str(soup)[:5000].lower()
                if "captcha-delivery.com" in body_str or "datadome" in body_str:
                    is_captcha_page = True
            if is_captcha_page:
                html = None  # treat as if HTML fetch failed, skip to fallbacks
                soup = None

        # Apply all HTML-based strategies
        if html and soup:
            strategies = [
                ("json_ld", extract_from_json_ld),
                ("link_tag", extract_from_link_tags),
                ("meta_tag", extract_from_meta_tags),
                ("img_tag", extract_from_img_tags),
                ("svg_fallback", extract_from_svg_inline),
                ("css_background", extract_from_css_background),
            ]

            for name, func in strategies:
                try:
                    if name == "img_tag":
                        found = func(soup, base_url, domain=domain)
                    else:
                        found = func(soup, base_url)
                    for url, score in found:
                        all_candidates.append((url, score, name))
                except Exception:
                    continue

            # Manifest (async)
            try:
                manifest_results = await extract_from_manifest(session, soup, base_url, domain)
                for url, score in manifest_results:
                    all_candidates.append((url, score, "manifest"))
            except Exception:
                pass

        # ------- Step 2b: Try direct /favicon.ico -------
        for scheme in ["https", "http"]:
            fav_url = f"{scheme}://{domain}/favicon.ico"
            try:
                async with session.get(
                    fav_url,
                    timeout=aiohttp.ClientTimeout(total=8),
                    ssl=False,
                    allow_redirects=True,
                ) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        if len(data) > MIN_IMAGE_SIZE:
                            all_candidates.append((fav_url, 30, "favicon_direct"))
                            break
            except Exception:
                continue

        # ------- Step 2c: WP REST API (for 403/blocked sites) -------
        # Only try WordPress REST API when HTML fetch failed - avoids extra
        # requests on sites that already loaded fine
        if not html:
            try:
                wp_results = await try_wp_json_logo(session, domain)
                all_candidates.extend(wp_results)
            except Exception:
                pass

        # ------- Step 3: API-based fallbacks (always tried) -------
        # Clearbit (high quality logos) - score 85 beats most HTML-extracted logos
        # because Clearbit provides clean, high-res company logos
        clearbit_url = await try_clearbit_logo(session, domain)
        if clearbit_url:
            all_candidates.append((clearbit_url, 85, "clearbit"))

        # Google favicon - always try as a reliable fallback
        google_url = await try_google_favicon(session, domain)
        if google_url:
            all_candidates.append((google_url, 25, "google_favicon"))

        # DuckDuckGo icon - secondary fallback
        if not all_candidates:
            ddg_url = await try_duckduckgo_icon(session, domain)
            if ddg_url:
                all_candidates.append((ddg_url, 20, "duckduckgo_icon"))

        # ------- Step 4: Pick best candidate -------
        if all_candidates:
            # Deduplicate by URL
            seen = set()
            unique = []
            for url, score, strategy in all_candidates:
                if url not in seen:
                    seen.add(url)
                    unique.append((url, score, strategy))

            # Sort by score descending
            unique.sort(key=lambda x: x[1], reverse=True)
            best_url, best_score, best_strategy = unique[0]

            result.logo_url = best_url
            result.source_strategy = best_strategy
            result._best_score = best_score

            # ------- Step 5: Download the image -------
            if DOWNLOAD_IMAGES:
                saved = await download_logo(session, best_url, domain, output_dir)
                if saved:
                    result.downloaded_path = saved
                elif len(unique) > 1:
                    # Try alternatives (up to 6)
                    for url2, _, strategy2 in unique[1:7]:
                        saved = await download_logo(session, url2, domain, output_dir)
                        if saved:
                            result.logo_url = url2
                            result.source_strategy = strategy2
                            result.downloaded_path = saved
                            break

                # ------- Step 5b: Google Favicon last resort -------
                if not result.downloaded_path:
                    gf = await download_google_favicon_fallback(
                        session, domain, output_dir)
                    if gf:
                        result.logo_url = f"https://www.google.com/s2/favicons?domain={domain}&sz=128"
                        result.source_strategy = "google_favicon_fallback"
                        result.downloaded_path = gf
        else:
            # ------- No candidates at all: try Google Favicon directly -------
            if DOWNLOAD_IMAGES:
                gf = await download_google_favicon_fallback(
                    session, domain, output_dir)
                if gf:
                    result.logo_url = f"https://www.google.com/s2/favicons?domain={domain}&sz=128"
                    result.source_strategy = "google_favicon_fallback"
                    result.downloaded_path = gf
                    result.error = None
                else:
                    result.error = "no_logo_found"
            else:
                result.error = "no_logo_found"

        result.extraction_time_ms = int((time.monotonic() - start) * 1000)
        return result


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

async def run_extraction(domains,output_dir,concurrency = CONCURRENCY):

    output_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)

    connector = aiohttp.TCPConnector(
        limit=concurrency,
        ttl_dns_cache=300,
        force_close=False,
        enable_cleanup_closed=True,
    )

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.ensure_future(
                extract_logo_for_domain(session, domain, output_dir, semaphore)
            )
            for domain in domains
        ]

        results = []
        total = len(tasks)
        completed = 0
        success = 0
        downloaded = 0
        start_time = time.monotonic()
        last_log = 0  # seconds since last log
        log_interval = 5  # log every N seconds

        for coro in asyncio.as_completed(tasks):
            try:
                r = await coro
                results.append(r)
                if r.logo_url:
                    success += 1
                if r.downloaded_path:
                    downloaded += 1
            except Exception as e:
                log.error(f"Task exception: {e}")

            completed += 1
            elapsed = time.monotonic() - start_time

            # Log progress every log_interval seconds or at 100% / every 200
            if elapsed - last_log >= log_interval or completed == total or completed % 200 == 0:
                last_log = elapsed
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate if rate > 0 else 0
                pct = success * 100 / max(completed, 1)
                bar_len = 30
                filled = int(bar_len * completed / total)
                bar = '█' * filled + '░' * (bar_len - filled)
                log.info(
                    f"{bar} {completed:>4d}/{total} ({completed*100//total:>2d}%) | "
                    f"Found: {success} ({pct:.1f}%) | "
                    f"DL: {downloaded} | "
                    f"{rate:.1f} d/s | "
                    f"ETA: {int(eta//60)}m{int(eta%60):02d}s"
                )

    return results


# ---------------------------------------------------------------------------
# Playwright dynamic extraction for SPA/JS-rendered sites
# ---------------------------------------------------------------------------

def _pw_extract_logo_from_page(page, domain):
    """
    Use Playwright to render a page and extract the logo.
    Looks for <img> elements in the top portion of the viewport
    that have logo-related attributes.
    Returns dict with logo_url and strategy, or None.
    """
    try:
        # Navigate with a generous timeout
        url = f"https://{domain}"
        try:
            page.goto(url, wait_until="networkidle", timeout=20000)
        except Exception:
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=15000)
            except Exception:
                # Try with www
                url = f"https://www.{domain}"
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=15000)
                except Exception:
                    return None

        # Wait a bit for lazy-loaded content
        page.wait_for_timeout(2000)

        # Detect captcha/challenge pages in Playwright
        pw_title = page.title().lower()
        captcha_signals = ["captcha", "security check", "challenge",
                           "just a moment", "attention required",
                           "ddos-guard", "access denied", "bot verification"]
        if any(s in pw_title for s in captcha_signals):
            return None
        # Also check for captcha-delivery.com in page content
        try:
            has_captcha = page.evaluate(r"""() => {
                const html = document.documentElement.innerHTML.substring(0, 5000).toLowerCase();
                return html.includes('captcha-delivery.com') || html.includes('datadome');
            }""")
            if has_captcha:
                return None
        except Exception:
            pass

        # Use JavaScript to find logo candidates in the rendered DOM
        candidates = page.evaluate(r"""() => {
            const results = [];
            
            function getClassName(el) {
                if (!el) return '';
                if (typeof el.className === 'string') return el.className;
                if (el.className && el.className.baseVal !== undefined) return el.className.baseVal;
                return '';
            }

            // Strategy 1: Find img elements with logo in class/id/alt/src
            const imgs = document.querySelectorAll('img');
            imgs.forEach((img, idx) => {
                let src = img.src || img.dataset?.src || '';
                // Fallback to srcset if no src
                if ((!src || src.startsWith('data:')) && img.srcset) {
                    const first = img.srcset.split(',')[0].trim().split(/\\s+/)[0];
                    if (first) src = first;
                }
                if (!src || src.startsWith('data:')) return;
                
                // Reject third-party cookie/consent/widget logos
                const rejectDomains = ['cookielaw.org', 'cookiebot.com', 'onetrust.com',
                    'cookie-cdn', 'cookie-script', 'trustarc', 'evidon.com',
                    'quantcast', 'termly.io', 'powered_by_logo', 'gdpr', 'consent-manager',
                    'gstatic.com/images/branding', 'google.com/images/branding',
                    'googlelogo', 'google-logo', 'translate.google', 'recaptcha.net',
                    'facebook.com/tr', 'analytics.', 'pixel.', 'beacon.',
                    'captcha-delivery.com', '/captcha/', 'datadome.co', 'hcaptcha.com'];
                if (rejectDomains.some(d => src.toLowerCase().includes(d))) return;

                const alt = (img.alt || '').toLowerCase();
                const cls = getClassName(img).toLowerCase();
                const id = (img.id || '').toLowerCase();
                const parentCls = getClassName(img.parentElement).toLowerCase();
                const parentId = (img.parentElement?.id || '').toLowerCase();
                const gpCls = getClassName(img.parentElement?.parentElement).toLowerCase();
                const gpId = (img.parentElement?.parentElement?.id || '').toLowerCase();

                const context = `${src} ${alt} ${cls} ${id} ${parentCls} ${parentId} ${gpCls} ${gpId}`.toLowerCase();

                // Check if it's in header/nav
                let inHeader = false, inNav = false, inFooter = false;
                let el = img;
                while (el) {
                    const tn = el.tagName;
                    const ecls = getClassName(el).toLowerCase();
                    const erole = (el.getAttribute && el.getAttribute('role') || '').toLowerCase();
                    if (tn === 'HEADER' || erole === 'banner') inHeader = true;
                    if (tn === 'NAV' || erole === 'navigation' || ecls.includes('navigation') || ecls.includes('navbar')) inNav = true;
                    if (tn === 'FOOTER') inFooter = true;
                    el = el.parentElement;
                }

                // Check position - logos are in the top of the page
                const rect = img.getBoundingClientRect();
                const inTopArea = rect.top < 200;

                // Calculate score
                let score = 0;
                if (context.includes('logo')) score += 50;
                if (context.includes('brand')) score += 30;
                if (context.includes('nav_logo') || context.includes('nav-logo') || context.includes('navigation-logo') || context.includes('navigation_logo')) score += 80;
                if (context.includes('home-logolink') || context.includes('home-logo') || context.includes('home_logo')) score += 80;
                if (context.includes('site-logo') || context.includes('site_logo')) score += 70;
                if (context.includes('header-logo') || context.includes('header_logo')) score += 60;
                if (cls.includes('logo') || id.includes('logo')) score += 40;
                if (parentCls === 'logo' || parentId === 'logo') score += 60;
                if (inHeader) score += 30;
                if (inNav) score += 20;
                if (inTopArea) score += 25;
                if (inFooter) score -= 30;
                
                // Homepage-link bonus: img inside <a href="/">
                const imgLink = img.closest('a');
                if (imgLink) {
                    const lhref = (imgLink.getAttribute('href') || '').trim();
                    const isHome = lhref === '/' || lhref === './' ||
                        lhref === location.origin || lhref === location.origin + '/';
                    if (isHome) score += 40;
                }
                
                if (context.includes('dealer')) score -= 20;
                if (context.includes('partner') || context.includes('sponsor')) score -= 25;
                if (src.endsWith('.svg')) score += 15;
                if (idx < 5) score += 10;

                // Visibility check
                const style = window.getComputedStyle(img);
                const visible = style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0';
                if (!visible) score -= 50;

                // ANY img in top area with non-trivial size gets a baseline
                if (inTopArea && visible && img.naturalWidth > 30 && img.naturalHeight > 10) {
                    score += 5;
                }

                // Penalize very large images (carousel/hero) with no logo signal
                if (img.naturalWidth > 600 && img.naturalHeight > 300 && score < 50) {
                    score -= 30;
                }

                // Reject third-party (off-domain) images with no logo signal
                try {
                    const imgHost = new URL(src).hostname;
                    const pageDomain = location.hostname.replace('www.', '');
                    if (!imgHost.includes(pageDomain) && !pageDomain.includes(imgHost.replace('www.',''))) {
                        if (score < 50) return; // off-domain + no logo signal = skip
                    }
                } catch(e) {}

                if (score > 10) {
                    results.push({url: src, score: score, strategy: 'playwright_img'});
                }
            });

            // Strategy 2: SVG elements that look like logos
            document.querySelectorAll('svg').forEach(svg => {
                // Skip tiny SVGs (UI icons like chevrons, arrows)
                const svgCls = getClassName(svg).toLowerCase();
                const iconHints = ['icon', 'toggle', 'menu', 'chevron', 'arrow', 'spinner', 'caret', 'close'];
                if (iconHints.some(h => svgCls.includes(h))) return;
                const ariaHidden = svg.getAttribute('aria-hidden');
                
                let el = svg;
                let inHeader = false, inNav = false;
                let parentHasLogo = false;
                while (el) {
                    const cls = getClassName(el).toLowerCase();
                    const id = (el.id || '').toLowerCase();
                    const role = (el.getAttribute && el.getAttribute('role') || '').toLowerCase();
                    if (el.tagName === 'HEADER' || cls.includes('header') || role === 'banner') inHeader = true;
                    if (el.tagName === 'NAV' || /\bnav[-\s]/.test(cls) || role === 'navigation' || cls.includes('navbar') || cls.includes('navigation')) inNav = true;
                    if (cls.includes('logo') || id.includes('logo') || cls.includes('brand') || id.includes('brand')) parentHasLogo = true;
                    el = el.parentElement;
                }
                
                function serializeSvg(score, strat) {
                    const link = svg.closest('a');
                    const img = link?.querySelector('img');
                    if (img && img.src) {
                        results.push({url: img.src, score: score, strategy: strat || 'playwright_svg_fallback'});
                    } else {
                        // Reject tiny SVG icons (close buttons, arrows, etc.)
                        const vb = svg.getAttribute('viewBox');
                        if (vb) {
                            const parts = vb.trim().split(/[\s,]+/);
                            if (parts.length >= 4) {
                                const vw = parseFloat(parts[2]), vh = parseFloat(parts[3]);
                                if (vw < 20 && vh < 20) return;
                            }
                        }
                        const ew = parseFloat(svg.getAttribute('width') || '0');
                        const eh = parseFloat(svg.getAttribute('height') || '0');
                        if (ew > 0 && eh > 0 && ew < 20 && eh < 20) return;
                        let svgStr = svg.outerHTML;
                        // Replace currentcolor with black so SVG renders standalone
                        svgStr = svgStr.replace(/fill\s*=\s*["']currentcolor["']/gi, 'fill="#000000"');
                        svgStr = svgStr.replace(/fill\s*:\s*currentcolor/gi, 'fill:#000000');
                        svgStr = svgStr.replace(/stroke\s*=\s*["']currentcolor["']/gi, 'stroke="#000000"');
                        svgStr = svgStr.replace(/stroke\s*:\s*currentcolor/gi, 'stroke:#000000');
                        if (svgStr.length > 200) {
                            const encoded = btoa(unescape(encodeURIComponent(svgStr)));
                            const dataUrl = 'data:image/svg+xml;base64,' + encoded;
                            results.push({url: dataUrl, score: score, strategy: strat || 'playwright_inline_svg'});
                        }
                    }
                }
                
                if (parentHasLogo && (inHeader || inNav)) {
                    serializeSvg(95, 'playwright_inline_svg');
                    return;
                }
                
                // Fallback: SVG inside <a> linking to homepage (even without logo class)
                // Catches sites like Airbnb with atomic CSS (no semantic class names)
                const link = svg.closest('a');
                if (link) {
                    const href = (link.getAttribute('href') || '').trim();
                    const ariaLabel = (link.getAttribute('aria-label') || '').toLowerCase();
                    const isHomepage = href === '/' || href === './' ||
                        href === location.origin || href === location.origin + '/' ||
                        (href.startsWith('http') && new URL(href, location.origin).pathname === '/');
                    if (isHomepage) {
                        const rect = svg.getBoundingClientRect();
                        if (inHeader || inNav || rect.top < 200) {
                            serializeSvg(85, 'playwright_inline_svg');
                            return;
                        }
                    }
                }
            });

            // Strategy 3: CSS background-image logos
            document.querySelectorAll('[class*="logo"], [class*="Logo"], [class*="brand"], [class*="Brand"]').forEach(el => {
                const style = window.getComputedStyle(el);
                const bg = style.backgroundImage || '';
                const match = bg.match(/url\(["']?([^"')]+)["']?\)/);
                if (!match) return;
                const bgUrl = match[1];
                if (!bgUrl || bgUrl.startsWith('data:') || bgUrl.includes('gradient')) return;
                // Must be an image file
                if (!/\.(svg|png|jpg|jpeg|gif|webp)/i.test(bgUrl)) return;
                const cls = getClassName(el).toLowerCase();
                let score = 70;
                if (cls.includes('logo')) score += 20;
                if (cls.includes('brand')) score += 10;
                // Check position
                const rect = el.getBoundingClientRect();
                if (rect.top < 200) score += 15;
                let inHeader = false;
                let parent = el;
                while (parent) {
                    if (parent.tagName === 'HEADER' || parent.tagName === 'NAV') { inHeader = true; break; }
                    if (getClassName(parent).toLowerCase().includes('header')) { inHeader = true; break; }
                    parent = parent.parentElement;
                }
                if (inHeader) score += 10;
                if (el.tagName === 'FOOTER' || getClassName(el).includes('footer')) score -= 40;
                results.push({url: bgUrl, score: score, strategy: 'playwright_css_bg'});
            });

            // Strategy 4: JSON-LD in the rendered page
            document.querySelectorAll('script[type="application/ld+json"]').forEach(script => {
                try {
                    const data = JSON.parse(script.textContent);
                    const items = Array.isArray(data) ? data : [data];
                    items.forEach(item => {
                        if (item.logo) {
                            const logoUrl = typeof item.logo === 'string' ? item.logo :
                                           (item.logo.url || item.logo.contentUrl || '');
                            if (logoUrl) {
                                // Penalize dealer logos
                                const lower = logoUrl.toLowerCase();
                                const dealerHints = ['dealer-logo', 'dealer_logo', '/dealer/',
                                    'dealer-logos', 'dealer_logos', 'haendler', 'franchise'];
                                const isDealerLogo = dealerHints.some(h => lower.includes(h));
                                results.push({url: logoUrl, score: isDealerLogo ? 50 : 90,
                                             strategy: 'playwright_jsonld'});
                            }
                        }
                    });
                } catch(e) {}
            });

            // Sort by score
            results.sort((a, b) => b.score - a.score);
            return results.slice(0, 5);
        }""")

        if candidates:
            best = candidates[0]
            return {"logo_url": best["url"], "strategy": best["strategy"]}
        return None

    except Exception as e:
        log.debug(f"Playwright error for {domain}: {e}")
        return None


def run_playwright_fallback(failed_domains, output_dir):
    """
    Process domains that failed static extraction using Playwright.
    Sequential processing (Playwright sync API is not thread-safe).
    """
    if not failed_domains:
        return {}

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log.warning("Playwright not installed - skipping dynamic fallback")
        return {}

    total = len(failed_domains)
    log.info(f"Playwright fallback: processing {total} domains...")
    results = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-extensions",
            ],
        )

        completed = 0
        found_count = 0

        for domain in failed_domains:
            start = time.monotonic()
            context = None
            page = None
            try:
                context = browser.new_context(
                    viewport={"width": 1280, "height": 720},
                    user_agent=USER_AGENTS[0],
                    ignore_https_errors=True,
                )
                context.set_default_timeout(15000)
                context.route("**/*.{mp4,webm,ogg,mp3,wav,flac,aac}",
                              lambda route: route.abort())
                context.route("**/*.{woff,woff2,ttf,eot}",
                              lambda route: route.abort())

                page = context.new_page()
                pw_result = _pw_extract_logo_from_page(page, domain)

                if pw_result and pw_result["logo_url"]:
                    result = LogoResult(
                        domain=domain,
                        logo_url=pw_result["logo_url"],
                        source_strategy=pw_result["strategy"],
                    )

                    if DOWNLOAD_IMAGES:
                        logo_url = pw_result["logo_url"]
                        if logo_url.startswith("data:image/svg+xml;base64,"):
                            import base64 as b64
                            try:
                                encoded = logo_url.split(",", 1)[1]
                                svg_data = _sanitize_svg(b64.b64decode(encoded))
                                if len(svg_data) > 100:
                                    filepath = output_dir / (sanitize_filename(domain) + ".svg")
                                    filepath.write_bytes(svg_data)
                                    result.downloaded_path = str(filepath)
                            except Exception:
                                pass
                        else:
                            import requests as req
                            import urllib3
                            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                            try:
                                resp = req.get(
                                    logo_url,
                                    timeout=10,
                                    verify=False,
                                    headers={"User-Agent": USER_AGENTS[0]},
                                )
                                if resp.status_code == 200 and len(resp.content) > MIN_IMAGE_SIZE:
                                    ct = resp.headers.get("Content-Type", "").lower()
                                    ext = ".png"
                                    if "svg" in ct: ext = ".svg"
                                    elif "jpeg" in ct or "jpg" in ct: ext = ".jpg"
                                    elif "webp" in ct: ext = ".webp"
                                    elif "gif" in ct: ext = ".gif"
                                    elif "ico" in ct: ext = ".ico"

                                    filepath = output_dir / (sanitize_filename(domain) + ext)
                                    content_to_write = resp.content
                                    if ext == ".svg" or "svg" in ct:
                                        content_to_write = _sanitize_svg(content_to_write)
                                    filepath.write_bytes(content_to_write)
                                    result.downloaded_path = str(filepath)
                            except Exception:
                                pass

                    result.extraction_time_ms = int(
                        (time.monotonic() - start) * 1000
                    )
                    results[domain] = result
                    found_count += 1
                    log.info(
                        f"  PW [{completed+1}/{total}] ✓ {domain}: "
                        f"{result.logo_url[:70]}"
                    )
                else:
                    log.info(
                        f"  PW [{completed+1}/{total}] ✗ {domain}: no logo"
                    )
            except Exception as e:
                log.debug(f"  PW error {domain}: {e}")
                log.info(
                    f"  PW [{completed+1}/{total}] ✗ {domain}: timeout/error"
                )
            finally:
                try:
                    if page:
                        page.close()
                    if context:
                        context.close()
                except Exception:
                    pass

            completed += 1
            if completed % 50 == 0:
                log.info(
                    f"  PW Progress: {completed}/{total} | "
                    f"Found: {found_count} "
                    f"({found_count*100//max(completed,1)}%)"
                )

        browser.close()

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def generate_report(results):
    """Print summary statistics and save results."""
    total = len(results)
    found = sum(1 for r in results if r.logo_url)
    downloaded = sum(1 for r in results if r.downloaded_path)
    errors = sum(1 for r in results if r.error)

    # Strategy breakdown
    strategy_counts = {}
    for r in results:
        if r.source_strategy:
            strategy_counts[r.source_strategy] = (
                strategy_counts.get(r.source_strategy, 0) + 1
            )

    log.info("=" * 60)
    log.info("EXTRACTION REPORT")
    log.info("=" * 60)
    log.info(f"Total domains:       {total}")
    log.info(f"Logos found (URL):   {found} ({found * 100 / total:.1f}%)")
    log.info(f"Logos downloaded:    {downloaded} ({downloaded * 100 / total:.1f}%)")
    log.info(f"No logo found:       {errors}")
    log.info("")
    log.info("Strategy breakdown:")
    for strategy, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
        log.info(f"  {strategy:20s}: {count:5d} ({count * 100 / found:.1f}%)")
    log.info("=" * 60)

    # Save JSON
    results_data = [asdict(r) for r in results]
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    log.info(f"Results saved to {RESULTS_FILE}")

    # Save CSV
    df = pd.DataFrame(results_data)
    df.to_csv(RESULTS_CSV, index=False)
    log.info(f"CSV saved to {RESULTS_CSV}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Entry point."""
    parquet_path = Path("logos.snappy.parquet")
    if not parquet_path.exists():
        log.error(f"Parquet file not found: {parquet_path}")
        sys.exit(1)

    log.info(f"Loading domains from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    domains = df["domain"].dropna().unique().tolist()
    log.info(f"Loaded {len(domains)} unique domains")

    # Optional: limit for testing
    if "--test" in sys.argv:
        n = 30
        domains = domains[:n]
        log.info(f"TEST MODE: processing first {n} domains only")

    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        n = int(sys.argv[idx + 1])
        domains = domains[:n]
        log.info(f"LIMIT MODE: processing first {n} domains only")

    log.info(f"Starting extraction with concurrency={CONCURRENCY}...")
    start = time.time()

    results = asyncio.run(run_extraction(domains, OUTPUT_DIR))

    # ------- Playwright fallback for failed/weak extractions -------
    if USE_PLAYWRIGHT and "--no-playwright" not in sys.argv:
        # Identify domains that need re-processing
        failed_domains = []
        weak_domains = []
        for r in results:
            if r.error or not r.logo_url:
                failed_domains.append(r.domain)
            elif r.downloaded_path:
                # Check if downloaded file is suspiciously small (favicon-only)
                try:
                    fsize = os.path.getsize(r.downloaded_path)
                    # Weak: tiny file from a fallback strategy, likely just a favicon
                    is_fallback_strategy = r.source_strategy in (
                        "google_favicon", "duckduckgo_icon", "favicon_direct", "link_tag"
                    )
                    # Also weak: low-confidence strategies (content images, no logo signal)
                    is_low_confidence = r.source_strategy in (
                        "meta_tag", "css_background"
                    )
                    # Consider weak if:
                    # - small file (<2KB) from favicon/link strategies
                    # - file is an .ico regardless of size
                    # - low-confidence strategy (meta images often aren't logos)
                    if (is_fallback_strategy and fsize < 2000) or \
                       r.downloaded_path.endswith(".ico") or \
                       is_low_confidence:
                        weak_domains.append(r.domain)
                except OSError:
                    pass

        retry_domains = failed_domains + weak_domains
        if retry_domains:
            log.info(
                f"Playwright fallback: {len(failed_domains)} failed + "
                f"{len(weak_domains)} weak = {len(retry_domains)} domains to retry"
            )
            pw_results = run_playwright_fallback(retry_domains, OUTPUT_DIR)

            # Merge Playwright results back
            for i, r in enumerate(results):
                if r.domain in pw_results:
                    pw_r = pw_results[r.domain]
                    results[i] = pw_r
                    log.info(f"  Updated {r.domain} via Playwright: {pw_r.logo_url}")

    elapsed = time.time() - start
    log.info(f"Extraction completed in {elapsed:.1f}s")

    generate_report(results)

    # Quick stats
    found = sum(1 for r in results if r.logo_url)
    log.info(
        f"\nFinal: {found}/{len(results)} logos extracted "
        f"({found * 100 / len(results):.1f}%) in {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
