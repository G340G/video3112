"""
scraper.py
- Scrape public-domain / openly-licensed images (Wikimedia Commons) based on randomized keywords.
- Also pulls one "normal" RSS headline (for the weird element).
"""
from __future__ import annotations

import os
import re
import time
import random
import hashlib
from dataclasses import dataclass
from typing import List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
import feedparser


WIKI_API = "https://commons.wikimedia.org/w/api.php"
UA = "AnalogHorrorBot/1.0 (GitHub Actions; educational)"


KEYWORDS_LIMINAL = [
    "liminal space", "empty hallway", "abandoned office", "fluorescent corridor",
    "empty mall", "backrooms", "night school corridor", "waiting room", "motel hallway",
    "carpet pattern", "stairwell", "utility room"
]
KEYWORDS_TECH_90S = [
    "CRT monitor", "camcorder", "VHS tape", "dial-up modem", "computer lab 1990s",
    "server room", "fax machine", "floppy disk", "dot matrix printer", "cathode ray",
    "answering machine"
]
KEYWORDS_CREEPY = [
    "fog", "abandoned", "ruins", "warning sign", "surveillance camera", "emergency exit",
    "maintenance", "electrical room", "hospital corridor"
]

RSS_SOURCES = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "https://www.theguardian.com/world/rss",
    "https://www.reuters.com/rssFeed/topNews",
]


@dataclass
class ScrapeConfig:
    out_dir: str
    max_images: int = 14
    min_width: int = 960
    min_height: int = 720
    allow_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
    seed: Optional[int] = None


def pick_keywords(seed: Optional[int] = None) -> List[str]:
    rng = random.Random(seed)
    pools = [KEYWORDS_LIMINAL, KEYWORDS_TECH_90S, KEYWORDS_CREEPY]
    kws = []
    for _ in range(rng.randint(2, 4)):
        x = rng.random()
        if x < 0.45:
            kws.append(rng.choice(pools[0]))
        elif x < 0.80:
            kws.append(rng.choice(pools[1]))
        else:
            kws.append(rng.choice(pools[2]))
    if rng.random() < 0.5:
        kws.append(rng.choice(["archive", "photograph", "1990", "security", "empty"]))
    out = []
    for k in kws:
        if k not in out:
            out.append(k)
    return out


def _safe_filename(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", s)
    return s[:120].strip("_") or "image"


def _download(url: str, dst: str, timeout: int = 35) -> bool:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": UA}, stream=True)
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
        return os.path.getsize(dst) > 10_000
    except Exception:
        return False


def _wikimedia_search_images(query: str, limit: int, rng: random.Random) -> List[dict]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|size|mime",
        "iiurlwidth": "1600",
    }
    r = requests.get(WIKI_API, params=params, headers={"User-Agent": UA}, timeout=35)
    r.raise_for_status()
    data = r.json()
    pages = (data.get("query", {}) or {}).get("pages", {}) or {}
    results = []
    for _pid, page in pages.items():
        iis = (page.get("imageinfo") or [])
        if not iis:
            continue
        ii = iis[0]
        url = ii.get("url")
        if not url:
            continue
        results.append({
            "title": page.get("title", ""),
            "url": url,
            "width": ii.get("width", 0),
            "height": ii.get("height", 0),
            "mime": ii.get("mime", ""),
        })
    rng.shuffle(results)
    return results


def _html_fallback_scrape_commons(query: str, rng: random.Random) -> List[str]:
    q = requests.utils.quote(query)
    url = f"https://commons.wikimedia.org/w/index.php?search={q}&title=Special:MediaSearch&type=image"
    r = requests.get(url, headers={"User-Agent": UA}, timeout=35)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    links = []
    for a in soup.select("a.sdms-image-result__thumbnail-link, a.mw-file-description"):
        href = a.get("href") or ""
        if href.startswith("/wiki/"):
            links.append("https://commons.wikimedia.org" + href)
    rng.shuffle(links)
    return links[:18]


def scrape_images(cfg: ScrapeConfig) -> List[str]:
    os.makedirs(cfg.out_dir, exist_ok=True)
    rng = random.Random(cfg.seed)

    keywords = pick_keywords(cfg.seed)
    query = " ".join(keywords)

    images: List[str] = []
    seen_urls = set()

    try:
        results = _wikimedia_search_images(query, limit=max(cfg.max_images * 3, 20), rng=rng)
    except Exception:
        results = []

    if not results:
        try:
            pages = _html_fallback_scrape_commons(query, rng=rng)
            titles = []
            for p in pages:
                m = re.search(r"/wiki/(File:[^?#]+)", p)
                if m:
                    titles.append(requests.utils.unquote(m.group(1)))
            if titles:
                params = {
                    "action": "query",
                    "format": "json",
                    "titles": "|".join(titles[:25]),
                    "prop": "imageinfo",
                    "iiprop": "url|size|mime",
                    "iiurlwidth": "1600",
                }
                rr = requests.get(WIKI_API, params=params, headers={"User-Agent": UA}, timeout=35)
                rr.raise_for_status()
                data = rr.json()
                pages2 = (data.get("query", {}) or {}).get("pages", {}) or {}
                for _pid, page in pages2.items():
                    iis = (page.get("imageinfo") or [])
                    if iis:
                        ii = iis[0]
                        url = ii.get("url")
                        if url:
                            results.append({
                                "title": page.get("title", ""),
                                "url": url,
                                "width": ii.get("width", 0),
                                "height": ii.get("height", 0),
                                "mime": ii.get("mime", ""),
                            })
                rng.shuffle(results)
        except Exception:
            results = []

    for item in results:
        if len(images) >= cfg.max_images:
            break
        url = item.get("url", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        w, h = int(item.get("width") or 0), int(item.get("height") or 0)
        if w < cfg.min_width or h < cfg.min_height:
            continue

        ext = os.path.splitext(url.split("?")[0])[1].lower()
        if ext not in cfg.allow_extensions:
            continue

        fn = _safe_filename(item.get("title") or hashlib.sha1(url.encode()).hexdigest()) + ext
        dst = os.path.join(cfg.out_dir, fn)
        if _download(url, dst):
            images.append(dst)

        time.sleep(0.1 + rng.random() * 0.15)

    return images


def fetch_normal_headline(seed: Optional[int] = None) -> str:
    rng = random.Random(seed)
    sources = RSS_SOURCES[:]
    rng.shuffle(sources)
    for src in sources:
        try:
            feed = feedparser.parse(src)
            if feed and feed.entries:
                entry = rng.choice(feed.entries[: min(15, len(feed.entries))])
                title = (entry.get("title") or "").strip()
                if title:
                    return re.sub(r"\s+", " ", title)
        except Exception:
            continue
    return "Local news update: nothing unusual reported."
