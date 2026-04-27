"""Web crawler: fetch website content and convert it to LangChain Documents."""

from __future__ import annotations

import logging
import re
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import httpx
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CompanyRAGBot/1.0; +https://company.internal)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "th,en;q=0.9",
}

_MAX_PAGES_PER_SOURCE = 50
_REQUEST_TIMEOUT = 30.0
_MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB


def _fetch_html(url: str, client: httpx.Client) -> Optional[str]:
    """Fetch a URL and return its HTML text, or None on failure."""
    try:
        resp = client.get(
            url,
            headers=_HEADERS,
            timeout=_REQUEST_TIMEOUT,
            follow_redirects=True,
        )
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "html" not in content_type and "text" not in content_type:
            logger.info("Skipping non-HTML URL: %s (content-type: %s)", url, content_type)
            return None
        if len(resp.content) > _MAX_CONTENT_LENGTH:
            logger.warning("Skipping oversized page: %s (%d bytes)", url, len(resp.content))
            return None
        return resp.text
    except httpx.HTTPStatusError as exc:
        logger.warning("HTTP %s fetching %s: %s", exc.response.status_code, url, exc)
        return None
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None


def _html_to_text(html: str, url: str) -> tuple[str, str]:
    """Parse HTML and return (title, plain_text). Falls back gracefully if bs4 missing."""
    try:
        from bs4 import BeautifulSoup  # noqa: PLC0415
    except ImportError:
        logger.error(
            "beautifulsoup4 is not installed. Run: pip install beautifulsoup4 lxml"
        )
        return url, ""

    try:
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "noscript", "iframe", "form", "button"]):
            tag.decompose()

        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        if not title:
            h1 = soup.find("h1")
            title = h1.get_text(strip=True) if h1 else url

        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return title, text.strip()
    except Exception as exc:
        logger.warning("HTML parsing failed for %s: %s", url, exc)
        return url, ""


def _extract_internal_links(html: str, base_url: str) -> List[str]:
    """Return internal links found in the HTML (same domain only)."""
    try:
        from bs4 import BeautifulSoup  # noqa: PLC0415
    except ImportError:
        return []

    try:
        soup = BeautifulSoup(html, "html.parser")
        base_parsed = urlparse(base_url)
        base_domain = base_parsed.netloc
        links: set[str] = set()

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
                continue
            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)
            if parsed.netloc == base_domain and parsed.scheme in ("http", "https"):
                clean = parsed._replace(fragment="").geturl()
                links.add(clean)

        return list(links)
    except Exception:
        return []


def crawl_url(url: str, crawl_depth: int = 0, source_name: str = "") -> List[Document]:
    """
    Fetch a website and return a list of LangChain Documents.

    Args:
        url: The starting URL to crawl.
        crawl_depth: 0 = single page only; 1 = also follow all internal links.
        source_name: Human-readable label stored in document metadata.

    Returns:
        List of Documents (one per crawled page).
    """
    docs: List[Document] = []
    visited: set[str] = set()
    to_visit: List[str] = [url]
    follow_links = crawl_depth >= 1

    with httpx.Client() as client:
        while to_visit and len(docs) < _MAX_PAGES_PER_SOURCE:
            current_url = to_visit.pop(0)
            if current_url in visited:
                continue
            visited.add(current_url)

            html = _fetch_html(current_url, client)
            if not html:
                continue

            title, text = _html_to_text(html, current_url)
            if not text.strip():
                logger.info("Empty text from %s — skipping.", current_url)
                continue

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": current_url,
                        "title": title,
                        "source_type": "url",
                        "source_name": source_name or url,
                    },
                )
            )
            logger.info("Crawled: %s — %d chars", current_url, len(text))

            if follow_links and current_url == url:
                links = _extract_internal_links(html, url)
                for link in links:
                    if link not in visited:
                        to_visit.append(link)

    logger.info("Total: %d page(s) crawled from %s", len(docs), url)
    return docs
