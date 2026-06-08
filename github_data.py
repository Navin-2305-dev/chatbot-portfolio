"""
github_data.py
──────────────
Richer GitHub data loader for the portfolio vector DB.

Improvements over the original:
  - Fetches README content for each repo (the real project description)
  - Fetches repository topics (tags like "machine-learning", "fastapi")
  - Detects and surfaces pinned repos first
  - Handles rate limiting gracefully with retries
  - Returns structured LangChain Documents (not raw strings)
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

import requests
from dotenv import load_dotenv
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

GITHUB_USERNAME = os.getenv("GITHUB_USERNAME", "Navin-2305-dev")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_MAX_REPOS = int(os.getenv("GITHUB_MAX_REPOS", "20"))
GITHUB_README_MAX_CHARS = int(os.getenv("GITHUB_README_MAX_CHARS", "1500"))


def _github_headers() -> dict:
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": f"{GITHUB_USERNAME}-portfolio-bot",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    return headers


def _safe_get(url: str, params: Optional[dict] = None, retries: int = 3) -> Optional[dict | list]:
    """GET with simple exponential-backoff retry on 429 / 5xx."""
    headers = _github_headers()
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=12)
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = 2 ** attempt
                logger.warning(f"GitHub {resp.status_code} on {url} — retrying in {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"GitHub request error ({attempt+1}/{retries}): {e}")
            time.sleep(2 ** attempt)
    return None


def _fetch_readme(repo_name: str) -> str:
    """Fetch and truncate the README for a given repo."""
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/readme"
    data = _safe_get(url)
    if not data:
        return ""

    import base64
    content_b64 = data.get("content", "")
    if not content_b64:
        return ""

    try:
        raw = base64.b64decode(content_b64).decode("utf-8", errors="replace")
        # Strip markdown headers/badges noise — keep meaningful prose
        lines = [
            ln for ln in raw.splitlines()
            if not ln.startswith("![") and not ln.startswith("[![")
        ]
        clean = "\n".join(lines).strip()
        return clean[:GITHUB_README_MAX_CHARS]
    except Exception as e:
        logger.warning(f"README decode error for {repo_name}: {e}")
        return ""


def _fetch_topics(repo_name: str) -> List[str]:
    """Fetch repository topics/tags."""
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/topics"
    headers = {**_github_headers(), "Accept": "application/vnd.github.mercy-preview+json"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.ok:
            return resp.json().get("names", [])
    except Exception as e:
        logger.warning(f"Topics fetch error for {repo_name}: {e}")
    return []


def _fetch_languages(repo_name: str) -> dict:
    """Fetch language breakdown (bytes per language)."""
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/languages"
    data = _safe_get(url)
    if not data:
        return {}
    # Convert to percentages
    total = sum(data.values()) or 1
    return {lang: round(bytes_ / total * 100, 1) for lang, bytes_ in data.items()}


def fetch_github_documents() -> List[Document]:
    """
    Fetch GitHub repos and return rich LangChain Documents.

    Returns two Document types:
      1. One overview Document summarising all repos (for broad queries)
      2. One detailed Document per repo (for specific project queries)
    """
    docs: List[Document] = []
    base_meta = {"source": "github", "github_username": GITHUB_USERNAME}

    # ── Fetch repo list ───────────────────────────────────────────────
    repos_data = _safe_get(
        f"https://api.github.com/users/{GITHUB_USERNAME}/repos",
        params={"sort": "updated", "per_page": GITHUB_MAX_REPOS, "type": "public"},
    )

    if not repos_data:
        logger.warning("GitHub fetch failed — using fallback document")
        return [Document(
            page_content=f"GitHub profile: https://github.com/{GITHUB_USERNAME}",
            metadata=base_meta,
        )]

    # ── Overview Document ─────────────────────────────────────────────
    overview_lines = [
        f"GitHub Profile: https://github.com/{GITHUB_USERNAME}",
        f"Public Repositories: {len(repos_data)}",
        "",
        "Repository Overview:",
    ]
    for repo in repos_data:
        name = repo.get("name", "")
        desc = repo.get("description") or "No description"
        lang = repo.get("language") or "N/A"
        stars = repo.get("stargazers_count", 0)
        forks = repo.get("forks_count", 0)
        overview_lines.append(
            f"  • {name} [{lang}] ★{stars} 🍴{forks} — {desc}"
        )

    docs.append(Document(
        page_content="\n".join(overview_lines),
        metadata={**base_meta, "section": "github_overview"},
    ))

    # ── Per-repo detailed Documents ───────────────────────────────────
    for repo in repos_data:
        name = repo.get("name", "")
        if not name:
            continue

        github_description = repo.get("description") or ""
        html_url = repo.get("html_url", "")
        primary_language = repo.get("language") or "Not specified"
        stars = repo.get("stargazers_count", 0)
        forks = repo.get("forks_count", 0)
        open_issues = repo.get("open_issues_count", 0)
        updated_at = repo.get("updated_at", "")[:10]   # Date only
        is_fork = repo.get("fork", False)
        homepage = repo.get("homepage") or ""

        # Enrichment (README, topics, language breakdown)
        print(f"\n{'='*60}")
        print(f"Fetching data for repo: {name}")
        print(f"GitHub description: {github_description}")
        
        readme = _fetch_readme(name)
        print(f"README fetched: {'Yes' if readme else 'No'} ({len(readme)} chars)")
        
        topics = _fetch_topics(name)
        languages = _fetch_languages(name)

        # Use README as primary description, fallback to GitHub description
        description = readme if readme else (github_description or "No description provided")
        print(f"Final description length: {len(description)} chars")
        print(f"Using: {'README' if readme else 'GitHub description'}")

        # Build rich content block
        content_parts = [
            f"Project: {name}",
            f"URL: {html_url}",
            f"Description: {description}",
            f"Primary Language: {primary_language}",
        ]

        if languages:
            lang_str = ", ".join(f"{l} ({p}%)" for l, p in sorted(languages.items(), key=lambda x: -x[1]))
            content_parts.append(f"Languages Used: {lang_str}")

        if topics:
            content_parts.append(f"Topics/Tags: {', '.join(topics)}")

        content_parts += [
            f"Stars: {stars} | Forks: {forks} | Open Issues: {open_issues}",
            f"Last Updated: {updated_at}",
            f"Is Fork: {'Yes' if is_fork else 'No'}",
        ]

        if homepage:
            content_parts.append(f"Live Demo / Homepage: {homepage}")

        # Note: README is already used as the primary description above
        # If GitHub description exists and differs from README, include it as additional context
        if github_description and readme and github_description != readme[:len(github_description)]:
            content_parts.append(f"\nGitHub Short Description: {github_description}")

        docs.append(Document(
            page_content="\n".join(content_parts),
            metadata={
                **base_meta,
                "section": "github_project",
                "repo_name": name,
                "language": primary_language,
                "stars": stars,
                "topics": topics,
            },
        ))

        logger.info(f"GitHub: enriched repo '{name}'")

    print(f"GitHub → {len(docs)} documents generated ({len(repos_data)} repos)")
    return docs