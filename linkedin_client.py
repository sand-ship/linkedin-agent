import logging
import os

import requests.cookies
from linkedin_api import Linkedin

import db

logger = logging.getLogger(__name__)

_client = None


def get_client() -> Linkedin:
    global _client
    if _client is None:
        li_at = os.environ.get("LINKEDIN_LI_AT", "").strip()
        if li_at:
            jar = requests.cookies.RequestsCookieJar()
            jar.set("li_at", li_at)
            jsessionid = os.environ.get("LINKEDIN_JSESSIONID", "").strip().strip('"')
            if jsessionid:
                jar.set("JSESSIONID", jsessionid)
            _client = Linkedin(
                "",
                "",
                cookies=jar,
                debug=False,
            )
        else:
            _client = Linkedin(
                os.environ["LINKEDIN_EMAIL"],
                os.environ["LINKEDIN_PASSWORD"],
                debug=False,
            )
    return _client


def _parse_degree(distance) -> int:
    if not distance:
        return 3
    val = distance.get("value", "") if isinstance(distance, dict) else str(distance)
    if "DISTANCE_1" in val or val == "F":
        return 1
    if "DISTANCE_2" in val or val == "S":
        return 2
    return 3


def _normalize(raw: dict, degree: int = None) -> dict:
    pub_id = raw.get("public_id") or raw.get("publicIdentifier") or ""
    urn_id = raw.get("urn_id") or raw.get("entityUrn", "").split(":")[-1] or pub_id

    return {
        "profile_id": urn_id,
        "name": raw.get("name") or f"{raw.get('firstName', '')} {raw.get('lastName', '')}".strip() or "Unknown",
        "headline": raw.get("jobtitle") or raw.get("headline") or raw.get("title") or "",
        "current_title": raw.get("jobtitle") or raw.get("headline") or raw.get("title") or "",
        "current_company": raw.get("company") or "",
        "location": raw.get("location") or raw.get("subline") or raw.get("locationName") or "",
        "industry": raw.get("industry") or "",
        "profile_url": f"https://linkedin.com/in/{pub_id}" if pub_id else "",
        "degree": degree if degree is not None else _parse_degree(raw.get("distance")),
    }


def _get_own_urn(api) -> str:
    """Get the current user's URN ID."""
    try:
        me = api.get_profile("me")
        urn = me.get("profile_id") or me.get("entityUrn", "").split(":")[-1]
        if urn:
            return urn
    except Exception as e:
        logger.warning(f"get_profile('me') failed: {e}")

    try:
        urn = api.client.metadata.get("urn_id", "")
        if urn:
            return urn.split(":")[-1]
    except Exception:
        pass

    return ""


def _save_batch(batch, count, degree=1):
    saved = 0
    for raw in batch:
        try:
            profile = _normalize(raw, degree=degree)
            if profile["profile_id"]:
                db.upsert_connection(profile)
                saved += 1
        except Exception as e:
            logger.warning(f"Skipping record: {e}")
    return saved


def sync_connections(progress_cb=None) -> int:
    """Pull all 1st-degree connections into local DB using two strategies:
    1. get_profile_connections (direct, bypasses search cap)
    2. Alphabet sweep via search_people (fallback to fill gaps)
    """
    api = get_client()
    logger.info("Starting connection sync...")

    if progress_cb:
        progress_cb("🔗 Logged into LinkedIn...")

    count = 0

    # --- Strategy 1: get_profile_connections (not capped by search limit) ---
    own_urn = _get_own_urn(api)
    if own_urn:
        logger.info(f"Own URN: {own_urn} — using get_profile_connections")
        if progress_cb:
            progress_cb("📥 Fetching via connections API...")
        try:
            connections = api.get_profile_connections(own_urn)
            count += _save_batch(connections, count)
            logger.info(f"get_profile_connections returned {len(connections)}, saved {count}")
            if progress_cb:
                progress_cb(f"💾 {count} connections fetched via connections API, now supplementing...")
        except Exception as e:
            logger.warning(f"get_profile_connections failed: {e}")

    # Build seen set from existing cache
    seen = set(c["profile_id"] for c in db.search_connections(limit=100000))
    logger.info(f"Starting sweeps with {len(seen)} already in cache")

    def sweep(terms, label):
        nonlocal count
        for i, term in enumerate(terms):
            try:
                batch = api.search_people(keywords=term, network_depths=["F"], limit=100)
                new = [r for r in batch if (r.get("urn_id") or "") not in seen]
                saved = _save_batch(new, count)
                count += saved
                for r in new:
                    seen.add(r.get("urn_id", ""))
                if saved:
                    logger.info(f"{label} '{term}': +{saved} new (total {count})")
                if progress_cb and i % 5 == 0:
                    progress_cb(f"💾 {label} sweep... {count} new so far ('{term}')")
            except Exception as e:
                logger.warning(f"{label} sweep failed for '{term}': {e}")

    sweep(list("abcdefghijklmnopqrstuvwxyz"), "Alphabet")

    locations = [
        "Bangalore", "Mumbai", "Delhi", "Hyderabad", "Chennai", "Pune",
        "Kolkata", "Gurgaon", "Noida", "Ahmedabad", "Singapore", "London",
        "San Francisco", "New York", "Dubai", "Seattle", "Boston", "Austin",
    ]
    sweep(locations, "Location")

    titles = [
        "founder", "CEO", "CTO", "director", "manager", "engineer",
        "product", "investor", "partner", "consultant", "analyst",
        "VP", "head", "lead", "principal", "architect",
    ]
    sweep(titles, "Title")

    logger.info(f"Sync complete — {count} new connections added")
    return count


def search_people(
    keywords: str = None,
    degrees: list = None,
    limit: int = 10,
    school: str = None,
    title: str = None,
    company: str = None,
):
    """Live LinkedIn people search. degrees = ['F'] and/or ['S']."""
    api = get_client()
    network_depths = degrees or ["F", "S"]

    kwargs = dict(
        network_depths=network_depths,
        limit=limit,
    )
    if keywords:
        kwargs["keywords"] = keywords
    if school:
        kwargs["keyword_school"] = school
    if title:
        kwargs["keyword_title"] = title
    if company:
        kwargs["keyword_company"] = company

    try:
        results = api.search_people(**kwargs)
        logger.info(f"Live search {kwargs} returned {len(results)} results")

        if not results:
            logger.info("Retrying without network_depths filter...")
            kwargs.pop("network_depths", None)
            results = api.search_people(**kwargs)
            logger.info(f"Retry returned {len(results)} results")

        if results:
            logger.info(f"First live result: {results[0]}")
        return [_normalize(r) for r in results]
    except Exception as e:
        logger.error(f"LinkedIn search failed: {e}")
        return []
