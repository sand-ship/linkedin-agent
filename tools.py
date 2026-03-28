import json
import logging
import os
from typing import Any

import httpx

import db
import embeddings
import linkedin_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schemas (passed to Claude)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "search_local_connections",
        "description": (
            "Search your locally cached 1st-degree LinkedIn connections. "
            "Fast — no LinkedIn requests. Always try this first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Title/headline keywords, e.g. ['founder', 'CEO', 'co-founder']",
                },
                "location": {
                    "type": "string",
                    "description": "Location filter, e.g. 'India', 'Bangalore'",
                },
                "company": {
                    "type": "string",
                    "description": "Company name filter",
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Broad keywords searched across ALL fields (name, headline, company). Use for company names like 'Amazon', 'Google' since current_company is often empty — the company name may appear in the headline instead.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 25)",
                    "default": 25,
                },
            },
        },
    },
    {
        "name": "search_linkedin_live",
        "description": (
            "Live LinkedIn people search. Slower; hits the LinkedIn API. "
            "Use when local cache has < 5 relevant results or user asks for 2nd-degree connections. "
            "Keep limit ≤ 15 to stay within rate limits."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "string",
                    "description": "Search string, e.g. 'SaaS founder India Series A'",
                },
                "include_second_degree": {
                    "type": "boolean",
                    "description": "Include 2nd-degree connections (default true)",
                    "default": True,
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results, keep ≤ 15",
                    "default": 10,
                },
            },
            "required": ["keywords"],
        },
    },
    {
        "name": "semantic_search_connections",
        "description": (
            "Semantic search over cached connections using embeddings. "
            "PREFER THIS over search_local_connections for any natural language query. "
            "Finds people based on meaning, not just keyword matches. "
            "'SaaS founder India' will match 'Building the Salesforce of B2B payments | Bangalore'. "
            "Returns results ranked by semantic similarity."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language description of who you're looking for, e.g. 'SaaS founders in India' or 'ex-Amazon people who started companies'",
                },
                "location": {
                    "type": "string",
                    "description": "Optional location filter applied after semantic search, e.g. 'India', 'Bangalore'",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 10)",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_cache_status",
        "description": "Returns how many connections are in the local cache and semantic index.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "enrich_company_funding",
        "description": (
            "Look up a company's latest funding round via Crunchbase. "
            "Only use when the user's query specifically mentions funding amounts or rounds. "
            "Returns funding amount, round type, and date."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company_name": {
                    "type": "string",
                    "description": "Company name to look up",
                }
            },
            "required": ["company_name"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def execute_tool(name: str, inputs: dict) -> Any:
    if name == "search_local_connections":
        results = db.search_connections(
            title_keywords=inputs.get("title_keywords"),
            location=inputs.get("location"),
            company=inputs.get("company"),
            keywords=inputs.get("keywords"),
            limit=inputs.get("limit", 25),
        )
        return {"count": len(results), "results": results}

    if name == "search_linkedin_live":
        degrees = ["F", "S"] if inputs.get("include_second_degree", True) else ["F"]
        results = linkedin_client.search_people(
            keywords=inputs["keywords"],
            degrees=degrees,
            limit=min(inputs.get("limit", 10), 15),
        )
        # Cache any new results we discover
        for r in results:
            if r.get("profile_id"):
                db.upsert_connection(r)
        return {"count": len(results), "results": results}

    if name == "semantic_search_connections":
        results = embeddings.semantic_search(
            query=inputs["query"],
            limit=inputs.get("limit", 10),
            location=inputs.get("location"),
        )
        return {"count": len(results), "results": results}

    if name == "get_cache_status":
        return {
            "cached_connections": db.get_connection_count(),
            "semantic_index_count": embeddings.index_count(),
        }

    if name == "enrich_company_funding":
        return _crunchbase_lookup(inputs["company_name"])

    return {"error": f"Unknown tool: {name}"}


def _crunchbase_lookup(company_name: str) -> dict:
    api_key = os.environ.get("CRUNCHBASE_API_KEY", "").strip()
    if not api_key:
        return {"error": "CRUNCHBASE_API_KEY not set — funding data unavailable"}

    # Check local cache first
    cached = db.get_enrichment(company_name)
    if cached:
        return cached

    try:
        url = "https://api.crunchbase.com/api/v4/searches/organizations"
        payload = {
            "field_ids": ["short_description", "funding_total", "last_funding_type", "last_funding_at"],
            "query": [{"type": "predicate", "field_id": "name", "operator_id": "eq", "values": [company_name]}],
            "limit": 1,
        }
        resp = httpx.post(url, json=payload, params={"user_key": api_key}, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        entities = data.get("entities", [])
        if not entities:
            return {"company": company_name, "found": False}

        props = entities[0].get("properties", {})
        funding_total = props.get("funding_total", {})
        amount = funding_total.get("value_usd") if isinstance(funding_total, dict) else None
        round_type = props.get("last_funding_type", "")
        date = props.get("last_funding_at", "")

        result = {
            "company": company_name,
            "found": True,
            "funding_amount_usd": amount,
            "last_round": round_type,
            "last_round_date": date,
        }
        if amount:
            db.upsert_enrichment(company_name, amount, round_type, date)
        return result

    except Exception as e:
        logger.error(f"Crunchbase lookup failed for {company_name}: {e}")
        return {"company": company_name, "error": str(e)}
