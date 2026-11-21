"""
AI Search Module - TRACE
-------------------------------------------------------------
Best-effort internet search to find possible reference versions of a video.

BUG FIX #11: Does not use random UUID filenames as search queries.
Results are labelled as "possible references", NOT as proof of authenticity.
Finding a matching video online does NOT prove the uploaded video is authentic.

Requires: requests, beautifulsoup4
-------------------------------------------------------------
"""

import re
from urllib.parse import quote


def search_real_video(query=None, source_context=None):
    """
    Search the internet for possible reference versions of a video.

    Args:
        query: A meaningful search query (title, person, event, source context).
               Should NOT be a random UUID filename.
        source_context: Optional context about where the video came from.

    Returns a list of dicts: {"title": ..., "url": ..., "note": ...}
    """
    results = []

    # Build a meaningful query. Do NOT use temporary filenames.
    search_query = source_context or query or ""
    search_query = search_query.strip()
    if not search_query:
        return [{
            "title": "No search query provided",
            "url": "#",
            "note": "Please provide a meaningful title, person, event, or source "
                    "context for the internet search.",
        }]

    # Clean the query a bit
    clean_query = re.sub(r"\.(mp4|avi|mov|mkv|webm)$", "", search_query,
                         flags=re.IGNORECASE)
    clean_query = clean_query.replace("_", " ").replace("-", " ")

    # Try DuckDuckGo HTML search (no API key needed)
    try:
        import requests
        from bs4 import BeautifulSoup

        url = "https://duckduckgo.com/html/?q=" + quote(clean_query + " original video")
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
                    "note": "Possible reference from web search - NOT proof of authenticity",
                })
    except Exception:
        results.append({
            "title": "Search could not run automatically",
            "url": "#",
            "note": "Try searching YouTube or Google manually for: " + clean_query,
        })

    # Always add a YouTube search link as a helpful option
    results.append({
        "title": "Search on YouTube",
        "url": "https://www.youtube.com/results?search_query=" + quote(clean_query),
        "note": "Check if an original version exists on YouTube - NOT proof of authenticity",
    })

    return results
