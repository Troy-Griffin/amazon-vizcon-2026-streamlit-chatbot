"""
CityPulse -- retrieval layer for the Gemini-backed chatbot (Tier 3).

Loads the same commuting data used by overview.html and globe-demo.html
(exported to city_data.json by build_streamlit_city_data.js) and pulls
out the real numbers for whichever city/metric a user question mentions.
This grounding context is handed to Gemini so it only has to phrase the
answer -- it should never need to invent a number.
"""
import difflib
import json
import re
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).parent / "city_data.json"

METRIC_INFO = {
    "affordability": "how cheap public transport fares are compared to other cities",
    "accessibility": "how dense the public transport network is (rail/BRT length, stations and lines per capita)",
    "adoption": "how widely public transport, cars and taxis are used relative to other cities",
    "efficiency": "how fast public transit moves relative to distance covered (average transit speed)",
    "variety": "how many different transport modes/services are available",
}

METRIC_ALIASES = {
    "affordability": ["afford", "cheap", "expensive", "fare", "cost"],
    "accessibility": ["access", "network density", "stations"],
    "adoption": ["adopt", "usage", "ridership"],
    "efficiency": ["efficient", "efficiency", "speed", "fast", "slow"],
    "variety": ["variety", "options", "diverse", "modes"],
}

NARRATIVE_PRIMER = (
    "CityPulse narrative context (for tone/reference, not per-city data): "
    "the site is built around the \"Marchetti constant\" -- the idea that people tend to accept "
    "roughly one hour of commuting a day (about 30 minutes each way), and that when transport "
    "gets faster, people tend to move further rather than bank the time saved. London is the "
    "site's headline example: covering about 1,587 km2, its commuters travel an estimated 39.7 "
    "million miles from home to work every day; doubled for the return trip over a five-day week, "
    "that is roughly the Earth-to-Sun-and-back distance. Historically, Rome packed over a million "
    "people into a radius of about 5 km, showing dense cities can outgrow what residents could "
    "traverse on foot once infrastructure supplies the resource base instead. CityPulse tracks "
    "five criteria per city -- affordability, accessibility, adoption, efficiency and variety -- "
    "combined into a weighted acceptance score (efficiency weighted highest, then accessibility, "
    "affordability, then variety)."
)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower())


class CityData:
    def __init__(self, data: dict[str, Any]):
        self.city_metrics: dict = data.get("cityMetrics", {})
        self.city_avg_commute: dict = data.get("cityAvgCommute", {})
        self.mode_share: dict = data.get("modeShareData", {})
        self.scatter_by_name = {d["name"]: d for d in data.get("scatterSpeedArea", [])}
        self.speed_time_dist_by_name = {d["name"]: d for d in data.get("speedTimeDistance", [])}
        self.norm_criteria_by_name = {d["name"]: d for d in data.get("normalizedCriteria", [])}
        self.density_by_name = {d["name"]: d for d in data.get("populationDensityCities", [])}
        self.london_deep_dive = data.get("londonDeepDive")
        self.criteria_weights = data.get("criteriaWeights")

        names = set()
        names.update(self.city_metrics.keys())
        names.update(self.city_avg_commute.keys())
        names.update(self.mode_share.keys())
        names.update(self.scatter_by_name.keys())
        names.update(self.density_by_name.keys())
        self.names = sorted(names)
        self._norm_names = {_norm(n): n for n in self.names}

    @classmethod
    def load(cls) -> "CityData":
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return cls(json.load(f))


_CACHE: CityData | None = None


def get_city_data() -> CityData:
    global _CACHE
    if _CACHE is None:
        _CACHE = CityData.load()
    return _CACHE


def find_city_matches(query: str, limit: int = 5) -> list[str]:
    idx = get_city_data()
    q = _norm(query)

    exact = [name for name in idx.names if _norm(name) in q]
    if exact:
        return exact[:limit]

    # Fuzzy fallback: check each 1-2 word window of the query against
    # known city names using difflib (typo tolerance, no extra deps).
    words = [w for w in q.split() if len(w) > 2]
    phrases = set(words)
    for i in range(len(words) - 1):
        phrases.add(words[i] + " " + words[i + 1])

    scored: dict[str, float] = {}
    norm_name_list = list(idx._norm_names.keys())
    for phrase in phrases:
        close = difflib.get_close_matches(phrase, norm_name_list, n=3, cutoff=0.72)
        for c in close:
            name = idx._norm_names[c]
            score = difflib.SequenceMatcher(None, phrase, c).ratio()
            if name not in scored or score > scored[name]:
                scored[name] = score

    ranked = sorted(scored.items(), key=lambda kv: -kv[1])
    return [name for name, _ in ranked[:limit]]


def find_metric_matches(query: str) -> list[str]:
    q = _norm(query)
    found = []
    for metric, aliases in METRIC_ALIASES.items():
        if metric in q or any(a in q for a in aliases):
            found.append(metric)
    return found


def _fmt(n, digits=1):
    if n is None:
        return "n/a"
    try:
        return f"{round(float(n), digits):,}"
    except (TypeError, ValueError):
        return str(n)


def format_city_block(name: str) -> str:
    idx = get_city_data()
    lines = [f"City: {name}"]

    m = idx.city_metrics.get(name)
    if m:
        lines.append(
            f"  Ranking scores (0-100 scale, normalized across ranked cities): "
            f"affordability {_fmt(m['affordability'])}, accessibility {_fmt(m['accessibility'])}, "
            f"adoption {_fmt(m['adoption'])}, efficiency {_fmt(m['efficiency'])}, variety {_fmt(m['variety'])}."
        )

    c = idx.city_avg_commute.get(name)
    if c:
        lines.append(
            f"  Real average commute (source: {c.get('source')}): distance {_fmt(c.get('meanDistanceKm'))} km. "
            f"By mode -- car {_fmt(c.get('meanCarMin'))} min, transit {_fmt(c.get('meanTransitMin'))} min, "
            f"bicycle {_fmt(c.get('meanBicycleMin'))} min, walking {_fmt(c.get('meanWalkingMin'))} min."
        )

    s = idx.scatter_by_name.get(name)
    if s:
        lines.append(
            f"  Functional urban area: {_fmt(s['areaKm2'], 0)} km2, population {_fmt(s['population'], 0)}, "
            f"mean transit speed {_fmt(s['speedKmh'])} km/h, dominant mode {s.get('dominantMode')}."
        )

    std = idx.speed_time_dist_by_name.get(name)
    if std:
        lines.append(
            f"  Mode-weighted average: {_fmt(std['timeMin'])} min commute, {_fmt(std['speedKmh'])} km/h, "
            f"{_fmt(std['distanceKm'])} km distance."
        )

    nc = idx.norm_criteria_by_name.get(name)
    if nc:
        lines.append(
            f"  Normalized criteria (0-100): affordability {_fmt(nc['affordability'])}, "
            f"accessibility {_fmt(nc['accessibility'])}, adoption {_fmt(nc['adoption'])}, "
            f"efficiency {_fmt(nc['efficiency'])}, variety {_fmt(nc['variety'])} -> "
            f"weighted acceptance score {_fmt(nc['weightedScore'])}."
        )

    d = idx.density_by_name.get(name)
    if d:
        lines.append(f"  Population density: {_fmt(d['populationDensityPerKm2'], 0)} people/km2 over {_fmt(d['areaKm2'], 0)} km2.")

    ms = idx.mode_share.get(name)
    if ms and ms.get("years"):
        years_str = " | ".join(
            f"{y['year']}: " + ", ".join(f"{mode} {pct}%" for mode, pct in y["shares"].items())
            for y in ms["years"]
        )
        lines.append(f"  Transport mode share over time -- {years_str}.")

    if name == "London" and idx.london_deep_dive:
        ld = idx.london_deep_dive
        lines.append(
            f"  London OD deep dive -- full scenario: {ld['full']['avgDistanceKm']} km / "
            f"{ld['full']['avgTimeMin']} min avg (n={ld['full']['lineCount']}); "
            f"medium+high density only: {ld['mediumHigh']['avgDistanceKm']} km / "
            f"{ld['mediumHigh']['avgTimeMin']} min (n={ld['mediumHigh']['lineCount']}); "
            f"high density only: {ld['highOnly']['avgDistanceKm']} km / "
            f"{ld['highOnly']['avgTimeMin']} min (n={ld['highOnly']['lineCount']}). "
            f"Region area {ld['regionAreaKm2']} km2."
        )

    if len(lines) == 1:
        return f"City: {name}\n  (No further ranking/commute data available for this city in the dataset.)"
    return "\n".join(lines)


def build_context(query: str) -> dict:
    """Returns {contextText, matchedCities, matchedMetrics} to ground a Gemini turn."""
    matched_cities = find_city_matches(query)
    matched_metrics = find_metric_matches(query)

    parts = [NARRATIVE_PRIMER]

    if matched_cities:
        parts.append(
            "\nDATA for the city/cities mentioned in the question "
            "(use these exact figures, do not invent or round differently):"
        )
        for name in matched_cities:
            parts.append(format_city_block(name))
    else:
        parts.append(
            "\nNo specific city was recognized in the question. If the user is asking about a "
            "city, politely ask them to name one of the cities covered by CityPulse (e.g. London, "
            "Tokyo, Berlin, New York, Paris, Chicago...). If the question is about the Marchetti "
            "constant or the general narrative, answer using the narrative context above."
        )

    if matched_metrics:
        parts.append("\nRelevant metric definitions:")
        for m in matched_metrics:
            parts.append(f"  {m}: {METRIC_INFO[m]}")

    return {
        "matchedCities": matched_cities,
        "matchedMetrics": matched_metrics,
        "contextText": "\n".join(parts),
    }
