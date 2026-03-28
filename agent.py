import asyncio
import json
import logging

import anthropic

from tools import TOOL_DEFINITIONS, execute_tool

logger = logging.getLogger(__name__)

_client = anthropic.AsyncAnthropic()

# In-memory conversation history keyed by Telegram chat_id
_conversations = {}  # chat_id -> list of messages

SYSTEM_PROMPT = """You are a LinkedIn search assistant with access to the user's cached connections and live LinkedIn search.

DATA REALITY — the cache only has 5 fields per person: name, headline, location, degree, urn_id.
- "SaaS", "B2B", industry labels almost never appear in headlines
- Company names sometimes appear in headlines (e.g. "Founder @ Razorpay") but not always
- School info is rarely in headlines

SEARCH STRATEGY — always run multiple searches, never give up after one:

For any query involving cached connections:
1. ALWAYS start with semantic_search_connections — it finds by meaning, not keywords.
   "SaaS founder India" will match "Building the future of B2B | Bangalore" or "CEO @ enterprise startup"
2. ALSO run search_linkedin_live — covers uncached 1st-degree + 2nd-degree connections.
   Live search results include actual degree (1st or 2nd) — never assume all live = 2nd degree.
3. Combine and deduplicate results from both, rank by: degree first, then similarity.

For school queries (Berkeley, IIT, Haas etc.):
- ALWAYS run TWO live searches in parallel: one with school="Berkeley", one with keywords="Berkeley India" (or whatever location)
- Also run semantic_search_connections for people who mention the school in their headline
- Combine and deduplicate all three result sets before responding
- Never give up after one search returns empty — the school filter and keyword search catch different people

For title queries (founder, VP, CTO etc.):
- Use search_linkedin_live with title="founder" instead of putting it in keywords

For company queries (Google, Sequoia etc.):
- Use search_linkedin_live with company="Google" for current employees
- For ex-employees, use keywords="ex-Google" or "formerly Google"

DATA NOTE: cache has name, headline, location only. Company/school only if in headline.

RESPONSE FORMAT:
- **Name** — Role | Location | 1st/2nd degree
- Show similarity score only if it helps explain a non-obvious match
- Single caveat line max — no suggestion lists
- NEVER ask permission — just search and show results
- Under 600 chars"""


async def run_agent(
    chat_id: int,
    user_message: str,
    progress_cb=None,
) -> str:
    history = _conversations.setdefault(chat_id, [])
    history.append({"role": "user", "content": user_message})

    messages = list(history)

    for _ in range(10):  # max 10 tool-call rounds
        response = await _client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

            async def _run_tool(block):
                if progress_cb:
                    await progress_cb(f"🔍 {_friendly_tool_name(block.name)}...")
                result = await asyncio.to_thread(execute_tool, block.name, block.input)
                return {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                }

            tool_results = list(await asyncio.gather(*[_run_tool(b) for b in tool_use_blocks]))
            tool_results = [_filter_results(tr) for tr in tool_results]

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        else:
            final_text = next(
                (b.text for b in response.content if hasattr(b, "text")),
                "Sorry, I couldn't generate a response.",
            )
            history.append({"role": "assistant", "content": final_text})
            # Keep history bounded
            if len(history) > 20:
                _conversations[chat_id] = history[-20:]
            return final_text

    return "Reached maximum tool call rounds. Please try a more specific query."


_ROLE_KEYWORDS      = {"founder", "ceo", "cto", "co-founder"}
_LOCATION_KEYWORDS  = {"india", "bangalore", "mumbai"}
_SENIORITY_KEYWORDS = {"series a", "yc", "ex-google"}
_SIMILARITY_THRESHOLD = 0.3


def _score_profile(profile: dict) -> float:
    score = 0.0
    headline = (profile.get("headline") or "").lower()
    location  = (profile.get("location") or "").lower()
    degree = profile.get("degree")

    if degree in (1, "F", "DISTANCE_1"):
        score += 3.0
    elif degree in (2, "S", "DISTANCE_2"):
        score += 1.0

    if any(kw in headline for kw in _ROLE_KEYWORDS):
        score += 2.0

    if any(kw in location for kw in _LOCATION_KEYWORDS):
        score += 1.0

    if any(kw in headline for kw in _SENIORITY_KEYWORDS):
        score += 1.0

    score += profile.get("similarity", 0.0)
    return score


def _filter_results(tool_result: dict) -> dict:
    """Score and filter profiles inside a tool_result dict in-place."""
    try:
        payload = json.loads(tool_result["content"])
    except (KeyError, json.JSONDecodeError):
        return tool_result

    profiles = payload.get("results")
    if not isinstance(profiles, list):
        return tool_result

    filtered = [
        p for p in profiles
        if p.get("similarity", 1.0) >= _SIMILARITY_THRESHOLD
    ]
    scored = sorted(filtered, key=_score_profile, reverse=True)

    payload["results"] = scored
    payload["count"] = len(scored)
    tool_result["content"] = json.dumps(payload)
    return tool_result


def clear_history(chat_id: int):
    _conversations.pop(chat_id, None)


def _friendly_tool_name(name: str) -> str:
    return {
        "search_local_connections": "Searching local cache",
        "search_linkedin_live": "Searching LinkedIn live",
        "batch_search_linkedin": "Batch searching LinkedIn",
        "get_cache_status": "Checking cache",
        "enrich_company_funding": "Looking up funding data",
    }.get(name, name)
