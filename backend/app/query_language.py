from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class AliasRule:
    canonical_name: str
    entity_id: str
    entity_type: str
    zh_terms: tuple[str, ...] = ()
    en_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class QueryLanguageProfile:
    query_language: str
    keywords: tuple[str, ...]
    exact_terms: tuple[str, ...]
    low_weight_terms: tuple[str, ...]
    concept_seeds: tuple[str, ...]
    expanded_terms: tuple[str, ...]
    bm25_query: str
    vector_query: str
    graph_seeds: tuple[str, ...]


ALIAS_RULES: tuple[AliasRule, ...] = (
    AliasRule(
        canonical_name="Unweighted Graph",
        entity_id="",
        entity_type="concept",
        zh_terms=("無權圖",),
        en_terms=("unweighted graph",),
    ),
    AliasRule(
        canonical_name="Shortest Path",
        entity_id="concept:shortest-path",
        entity_type="concept",
        zh_terms=("最短步數", "最短路徑"),
        en_terms=("shortest path", "shortest steps"),
    ),
    AliasRule(
        canonical_name="BFS",
        entity_id="concept:bfs",
        entity_type="algorithm",
        zh_terms=("廣度優先搜尋", "廣搜"),
        en_terms=("BFS", "breadth first search", "breadth-first search"),
    ),
    AliasRule(
        canonical_name="Queue",
        entity_id="concept:queue",
        entity_type="data_structure",
        zh_terms=("佇列", "隊列"),
        en_terms=("Queue",),
    ),
    AliasRule(
        canonical_name="Visited Array",
        entity_id="concept:visited-array",
        entity_type="technique",
        zh_terms=("拜訪陣列", "visited 陣列"),
        en_terms=("Visited Array", "visited array", "visited set"),
    ),
    AliasRule(
        canonical_name="Graph Traversal",
        entity_id="pattern:graph-traversal",
        entity_type="pattern",
        zh_terms=("圖論遍歷", "圖遍歷"),
        en_terms=("Graph Traversal", "graph traversal"),
    ),
    AliasRule(
        canonical_name="Source Target",
        entity_id="",
        entity_type="concept",
        zh_terms=("起點", "終點"),
        en_terms=("source", "target", "start", "end"),
    ),
    AliasRule(
        canonical_name="Grid",
        entity_id="",
        entity_type="concept",
        zh_terms=("網格",),
        en_terms=("grid", "matrix"),
    ),
)

LOW_WEIGHT_TERMS = (
    "給定",
    "請",
    "找出",
    "從",
    "到",
    "需要",
    "明確",
    "使用",
    "哪些",
)

ANALYSIS_SIGNAL_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("BFS", ("bfs", "breadth first", "breadth-first", "廣度優先搜尋", "廣搜")),
    ("Queue", ("queue", "deque", "佇列", "隊列")),
    ("Visited Array", ("visited", "vis[", "拜訪", "標記")),
    (
        "Unweighted shortest path",
        ("unweighted", "無權", "shortest path", "shortest steps", "最短", "minimum steps"),
    ),
    ("Graph", ("graph", "node", "edge", "vertex", "圖", "節點", "邊", "grid", "matrix", "網格")),
)

_ASCII_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")

GRAPH_SEED_ENTITY_REGISTRY = {
    rule.entity_id: {
        "entityId": rule.entity_id,
        "name": rule.canonical_name,
        "type": rule.entity_type,
        "entityType": rule.entity_type,
    }
    for rule in ALIAS_RULES
    if rule.entity_id
}


def build_query_language_profile(text: str) -> QueryLanguageProfile:
    normalized = " ".join(text.strip().split())
    exact_terms = extract_exact_terms(normalized)
    concept_seeds = infer_concept_seeds(normalized, exact_terms)
    expanded_terms = expand_terms(exact_terms, concept_seeds)
    graph_seed_names = _dedupe((*concept_seeds, *_matched_canonical_names(normalized, exact_terms)))
    graph_seeds = _graph_seed_entity_ids(graph_seed_names)
    return QueryLanguageProfile(
        query_language=detect_query_language(normalized),
        keywords=shared_multilingual_tokens(normalized),
        exact_terms=exact_terms,
        low_weight_terms=extract_low_weight_terms(normalized),
        concept_seeds=concept_seeds,
        expanded_terms=expanded_terms,
        bm25_query=build_bm25_query(normalized, expanded_terms),
        vector_query=build_vector_query(normalized, exact_terms, concept_seeds, expanded_terms),
        graph_seeds=graph_seeds,
    )


def detect_query_language(text: str) -> str:
    has_cjk = bool(_CJK_RE.search(text))
    has_ascii_letters = bool(_ASCII_LETTER_RE.search(text))
    if has_cjk and has_ascii_letters:
        return "mixed"
    if has_cjk:
        return "zh-Hant"
    return "en"


def extract_exact_terms(text: str) -> tuple[str, ...]:
    if not text:
        return ()

    exact_terms: list[str] = []
    for rule in ALIAS_RULES:
        for term in _ordered_terms(rule):
            if _contains_term(text, term):
                exact_terms.append(term)
    return _dedupe(exact_terms)


def extract_low_weight_terms(text: str) -> tuple[str, ...]:
    return tuple(term for term in LOW_WEIGHT_TERMS if term in text)


def _naked_concept_query_seed(
    text: str,
    matched_canonicals: tuple[str, ...],
) -> str | None:
    normalized_query = " ".join(text.strip().lower().split())
    if not normalized_query:
        return None

    for rule in ALIAS_RULES:
        if not rule.entity_id or rule.canonical_name not in matched_canonicals:
            continue
        if any(normalized_query == term.lower() for term in _ordered_terms(rule)):
            return rule.canonical_name
    return None


def infer_concept_seeds(text: str, exact_terms: tuple[str, ...]) -> tuple[str, ...]:
    matched_canonicals = _matched_canonical_names(text, exact_terms)
    exact_term_set = set(exact_terms)
    seeds: list[str] = []

    naked_concept_seed = _naked_concept_query_seed(text, matched_canonicals)
    if naked_concept_seed is not None:
        seeds.append(naked_concept_seed)

    if "BFS" in matched_canonicals:
        seeds.extend(("BFS", "Queue", "Visited Array"))
    if "Shortest Path" in matched_canonicals:
        seeds.append("Shortest Path")
    if "Unweighted Graph" in matched_canonicals and "Shortest Path" in matched_canonicals:
        seeds.extend(("BFS", "Shortest Path"))
    if "Grid" in matched_canonicals and "Shortest Path" in matched_canonicals:
        seeds.extend(("Graph Traversal", "BFS"))
    if {"起點", "終點"} <= exact_term_set and "Shortest Path" in matched_canonicals:
        seeds.extend(("Shortest Path", "BFS"))
    if _has_graph_context(text) and {"BFS", "Shortest Path"} & set(seeds):
        seeds.append("Graph Traversal")
    if "BFS" in seeds:
        seeds.extend(("Queue", "Visited Array"))

    return _dedupe(seeds)


def expand_terms(
    exact_terms: tuple[str, ...],
    concept_seeds: tuple[str, ...],
) -> tuple[str, ...]:
    matched_canonicals = set(_matched_canonical_names("", exact_terms))
    matched_canonicals.update(concept_seeds)

    expanded: list[str] = []
    for rule in ALIAS_RULES:
        if rule.canonical_name in matched_canonicals:
            expanded.extend(rule.en_terms)
    return _dedupe(expanded)


def shared_multilingual_tokens(text: str) -> tuple[str, ...]:
    exact_terms = extract_exact_terms(text)
    observed_phrase_terms = tuple(term for term in exact_terms if _CJK_RE.search(term))
    ascii_tokens = _ASCII_TOKEN_RE.findall(text.lower())
    return _dedupe((*observed_phrase_terms, *ascii_tokens))


def build_bm25_query(original_query: str, expanded_terms: tuple[str, ...]) -> str:
    return " ".join(_dedupe((original_query, *expanded_terms))).strip()


def concept_search_aliases(
    concepts: tuple[str, ...],
    problem_type: str = "",
) -> tuple[str, ...]:
    aliases: list[str] = []
    for name in (*concepts, *((problem_type,) if problem_type else ())):
        rule = _rule_for_canonical_name(name)
        if rule is None:
            continue
        for term in (*rule.zh_terms, *rule.en_terms):
            if term == name:
                continue
            aliases.append(term)
    return _dedupe(aliases)


def build_search_text(
    *,
    problem_id: str,
    source: str,
    source_id: str,
    title: str,
    problem_type: str,
    concepts: tuple[str, ...],
    display_text: str,
) -> str:
    parts = [
        problem_id,
        source,
        source_id,
        f"{source}-{source_id}" if source and source_id else "",
        f"{source} {source_id}" if source and source_id else "",
        title,
        problem_type,
        " ".join(concepts),
        " ".join(concept_search_aliases(concepts, problem_type)),
        display_text,
    ]
    return " ".join(part for part in parts if part).strip()


def build_vector_query(
    original_query: str,
    exact_terms: tuple[str, ...],
    concept_seeds: tuple[str, ...],
    expanded_terms: tuple[str, ...],
) -> str:
    matched_canonicals = _matched_canonical_names(original_query, exact_terms)
    if (
        "Unweighted Graph" in matched_canonicals
        and "Shortest Path" in concept_seeds
        and {"起點", "終點"} <= set(exact_terms)
    ):
        return "find the shortest path in an unweighted graph from source to target using bfs and a queue"
    if "Grid" in matched_canonicals and "Shortest Path" in concept_seeds:
        return "find the shortest path on a grid using bfs and a queue"
    if expanded_terms:
        return " ".join(expanded_terms)
    return original_query


def detect_traversal_signal_labels(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    labels: list[str] = []
    for label, markers in ANALYSIS_SIGNAL_MARKERS:
        if any(_contains_marker(text, lowered, marker) for marker in markers):
            labels.append(label)
    return tuple(labels)


def _graph_seed_entity_ids(concept_seeds: tuple[str, ...]) -> tuple[str, ...]:
    entity_ids: list[str] = []
    for concept in concept_seeds:
        rule = _rule_for_canonical_name(concept)
        if rule is None or not rule.entity_id:
            continue
        entity_ids.append(rule.entity_id)
    return _dedupe(entity_ids)


def _matched_canonical_names(text: str, exact_terms: tuple[str, ...]) -> tuple[str, ...]:
    matched: list[str] = []
    exact_term_set = set(exact_terms)
    for rule in ALIAS_RULES:
        if any(term in exact_term_set for term in (*rule.zh_terms, *rule.en_terms)):
            matched.append(rule.canonical_name)
            continue
        if text and any(_contains_term(text, term) for term in (*rule.zh_terms, *rule.en_terms)):
            matched.append(rule.canonical_name)
    return _dedupe(matched)


def _rule_for_canonical_name(name: str) -> AliasRule | None:
    for rule in ALIAS_RULES:
        if rule.canonical_name == name:
            return rule
    return None


def _ordered_terms(rule: AliasRule) -> tuple[str, ...]:
    return tuple(
        sorted(
            (*rule.zh_terms, *rule.en_terms),
            key=len,
            reverse=True,
        )
    )


def _contains_term(text: str, term: str) -> bool:
    if not term:
        return False
    if _CJK_RE.search(term):
        return term in text
    lowered = text.lower()
    return term.lower() in lowered


def _contains_marker(text: str, lowered: str, marker: str) -> bool:
    if _CJK_RE.search(marker):
        return marker in text
    return marker in lowered


def _has_graph_context(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in ("graph", "grid", "matrix")) or any(
        marker in text for marker in ("圖", "網格")
    )


def _dedupe(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))
