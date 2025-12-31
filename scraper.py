# scraper.py
from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import feedparser
import requests

USER_AGENT = "AnalogHorrorBot/1.0 (GitHub Actions; contact: none)"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


KEYWORDS = [
    "liminal space hallway",
    "abandoned office 1990s",
    "crt monitor closeup",
    "vhs cassette label",
    "dial-up modem",
    "computer lab 1990s",
    "fluorescent corridor",
    "empty mall interior",
    "security camera still",
    "analog television static",
    "office cubicles night",
    "stairwell institutional",
]

RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "https://www.theguardian.com/world/rss",
]


@dataclass
class ImageAsset:
    url: str
    source: str
    title: str
    license_hint: str = ""


def pick_keyword(seed: Optional[int] = None) -> str:
    rnd = random.Random(seed)
    return rnd.choice(KEYWORDS)


def _commons_search_files(query: str, limit: int = 12) -> List[str]:
    """
    Returns Commons file titles (e.g., 'File:Something.jpg') by searching.
    Uses MediaWiki API generator=search in namespace 6 (File).
    """
    api = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrnamespace": 6,
        "gsrsearch": query,
        "gsrlimit": limit,
        "origin": "*",
    }
    r = SESSION.get(api, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    pages = (data.get("query") or {}).get("pages") or {}
    titles = []
    for _, p in pages.items():
        t = p.get("title")
        if t and t.lower().startswith("file:"):
            titles.append(t)
    return titles


def _commons_file_info(file_title: str, thumb_px: int = 1600) -> Optional[ImageAsset]:
    """
    Fetches imageinfo + extmetadata to try to prefer Public Domain / CC0.
    Returns a direct URL (thumb) that is easy to download.
    """
    api = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": file_title,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": thumb_px,
        "origin": "*",
    }
    r = SESSION.get(api, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    pages = (data.get("query") or {}).get("pages") or {}
    for _, p in pages.items():
        iis = p.get("imageinfo") or []
        if not iis:
            continue
        ii = iis[0]
        thumb = ii.get("thumburl") or ii.get("url")
        meta = ii.get("extmetadata") or {}
        lic_short = (meta.get("LicenseShortName") or {}).get("value", "")
        lic_url = (meta.get("LicenseUrl") or {}).get("value", "")
        artist = (meta.get("Artist") or {}).get("value", "")
        title = (meta.get("ObjectName") or {}).get("value", file_title)

        license_hint = " ".join([lic_short, lic_url]).strip()

        # Heuristic: prefer PD / CC0 (not bulletproof).
        ok = False
        l = (lic_short or "").lower()
        if "public domain" in l or "cc0" in l or "pd" == l.strip():
            ok = True

        # Still allow if no metadata (Commons thumbnails are usually safe to fetch,
        # but licensing may vary; you can tighten this if needed).
        if ok or not lic_short:
            clean_title = re.sub(r"<.*?>", "", title).strip()
            clean_artist = re.sub(r"<.*?>", "", artist).strip()
            return ImageAsset(
                url=thumb,
                source="Wikimedia Commons",
                title=f"{clean_title} — {clean_artist}".strip(" —"),
                license_hint=license_hint,
            )
    return None


def fetch_commons_images(query: str, max_images: int = 6) -> List[ImageAsset]:
    titles = _commons_search_files(query, limit=max_images * 3)
    random.shuffle(titles)

    out: List[ImageAsset] = []
    for t in titles:
        asset = _commons_file_info(t)
        if asset and asset.url:
            out.append(asset)
        if len(out) >= max_images:
            break
    return out


def fetch_archive_images(query: str, max_images: int = 4) -> List[ImageAsset]:
    """
    Uses archive.org advancedsearch endpoint (JSON) to find image items and extract a candidate file URL.
    Heuristic: looks for identifiers and uses the /download/{identifier}/ URL pattern.
    """
    # Basic advancedsearch: mediatype:image and query terms
    endpoint = "https://archive.org/advancedsearch.php"
    q = f'({query}) AND mediatype:image'
    params = {
        "q": q,
        "fl[]": ["identifier", "title"],
        "rows": min(50, max_images * 10),
        "page": 1,
        "output": "json",
    }
    r = SESSION.get(endpoint, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    docs = (((data.get("response") or {}).get("docs")) or [])
    random.shuffle(docs)

    assets: List[ImageAsset] = []
    for d in docs:
        ident = d.get("identifier")
        title = d.get("title") or ident or "archive_item"
        if not ident:
            continue

        # This points to a directory listing; we’ll download via requests and pick the first suitable image later.
        url = f"https://archive.org/download/{ident}/"
        assets.append(ImageAsset(url=url, source="Archive.org", title=str(title), license_hint="(check item page)"))

        if len(assets) >= max_images:
            break
    return assets


def resolve_archive_download_dir(download_dir_url: str) -> Optional[str]:
    """
    Given https://archive.org/download/{identifier}/
    tries to find a direct .jpg/.png file link by parsing the directory listing HTML.
    """
    try:
        r = SESSION.get(download_dir_url, timeout=30)
        r.raise_for_status()
        html = r.text
    except Exception:
        return None

    # crude href parse (keeps dependencies minimal; bs4 would also work)
    hrefs = re.findall(r'href="([^"]+)"', html, flags=re.IGNORECASE)
    candidates = []
    for h in hrefs:
        if h.lower().endswith((".jpg", ".jpeg", ".png", ".webp")) and not h.startswith("?"):
            candidates.append(h)

    if not candidates:
        return None

    pick = random.choice(candidates)
    if pick.startswith("http"):
        return pick
    return download_dir_url.rstrip("/") + "/" + pick.lstrip("/")


def pick_rss_headline(seed: Optional[int] = None) -> str:
    rnd = random.Random(seed)
    feed_url = rnd.choice(RSS_FEEDS)
    try:
        d = feedparser.parse(feed_url)
        if not d.entries:
            return "Local services resume after minor disruption."
        e = rnd.choice(d.entries[: min(25, len(d.entries))])
        title = getattr(e, "title", None) or "Routine update issued by authorities."
        return re.sub(r"\s+", " ", title).strip()
    except Exception:
        return "Routine update issued by authorities."
