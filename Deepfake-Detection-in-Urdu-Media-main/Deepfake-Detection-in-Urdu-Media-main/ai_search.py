"""
AI Search Module
-------------------------------------------------------------
Best-effort internet search to find the "real" version of a video.

It tries to look up the video title/link on the web and returns
possible matches so the student can compare. This is a simple demo
- it does not guarantee finding the real video.

Requires: requests, beautifulsoup4
-------------------------------------------------------------
"""

import re
from urllib.parse import quote


def search_real_video(query):
    """
    Search the internet for the real version of a video.

    Returns a list of dicts: {"title": ..., "url": ..., "note": ...}
    """
    results = []

    # Clean the query a bit
    query = query.strip()
    if not query:
        return results

    # Remove file extensions for cleaner searching
    clean_query = re.sub(r"\.(mp4|avi|mov|mkv|webm)$", "", query, flags=re.IGNORECASE)
    clean_query = clean_query.replace("_", " ").replace("-", " ")

    # Try DuckDuckGo HTML search (no API key needed)
    try:
        import requests
        from bs4 import BeautifulSoup

        url = "https://duckduckgo.com/html/?q=" + quote(clean_query + " real video")
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        for res in soup.select(".result")[:5]:
            title_tag = res.select_one(".result__a")
            if title_tag:
                title = title_tag.get_text(strip=True)
                link = title_tag.get("href", "#")
                results.append({
                    "title": title,
                    "url": link,
                    "note": "Possible match from web search",
                })
    except Exception:
        # If search fails, return a helpful placeholder
        results.append({
            "title": "Search could not run automatically",
            "url": "#",
            "note": "Try searching YouTube or Google manually for: " + clean_query,
        })

    # Always add a YouTube search link as a helpful option
    results.append({
        "title": "Search on YouTube",
        "url": "https://www.youtube.com/results?search_query=" + quote(clean_query),
        "note": "Check if the original/real video exists on YouTube",
    })

    return results
