from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from ..contracts import EntityRecord, ProblemChunk, RawProblem, RelationRecord
from ..providers import DeterministicMockEmbeddingProvider, EmbeddingProvider
from ..stores import GraphStore, VectorRecord, VectorStore


Target = Literal["json", "bm25", "qdrant", "neo4j", "all"]


class IngestionError(RuntimeError):
    pass


def build_ingestion_artifacts(
    *,
    input_dir: Path,
    processed_dir: Path,
    target: Target = "all",
    allow_fallback: bool = False,
    embedding_provider: EmbeddingProvider | None = None,
    vector_store: VectorStore | None = None,
    graph_store: GraphStore | None = None,
) -> dict[str, Any]:
    if target not in {"json", "bm25", "qdrant", "neo4j", "all"}:
        raise IngestionError(f"unsupported target: {target}")

    problems = _load_raw_problems(input_dir)
    chunks = _build_chunks(problems)
    entities, relations = _extract_entities_and_relations(problems)
    processed_dir.mkdir(parents=True, exist_ok=True)

    _write_json(processed_dir / "problems.json", [problem.to_mapping() for problem in problems])
    _write_json(processed_dir / "chunks.json", [chunk.to_mapping() for chunk in chunks])
    _write_json(processed_dir / "entities.json", [entity.to_mapping() for entity in entities])
    _write_json(processed_dir / "relations.json", [relation.to_mapping() for relation in relations])

    fallback = {"qdrant": False, "neo4j": False}
    if target in {"bm25", "all"}:
        _write_bm25_index(processed_dir / "bm25_index.json", chunks)
    if target in {"qdrant", "all"}:
        vector_records, qdrant_payload = _build_qdrant_vectors(
            chunks,
            embedding_provider or DeterministicMockEmbeddingProvider(),
        )
        _write_json(processed_dir / "qdrant_vectors.json", qdrant_payload)
        if vector_store is not None:
            vector_store.upsert(vector_records)
        elif allow_fallback:
            fallback["qdrant"] = True
        else:
            try:
                from ..adapters.qdrant import QdrantVectorStore

                QdrantVectorStore().upsert(vector_records)
            except Exception as exc:
                raise IngestionError(
                    "Qdrant is not available; start Docker or pass --allow-fallback"
                ) from exc
    if target in {"neo4j", "all"}:
        _write_neo4j_graph(processed_dir / "neo4j_graph.json", entities, relations)
        if graph_store is not None:
            graph_store.upsert_entities(entities)
            graph_store.upsert_relations(relations)
        elif allow_fallback:
            fallback["neo4j"] = True
        else:
            try:
                from ..adapters.neo4j import Neo4jGraphStore

                store = Neo4jGraphStore()
                store.upsert_entities(entities)
                store.upsert_relations(relations)
            except Exception as exc:
                raise IngestionError(
                    "Neo4j is not available; start Docker or pass --allow-fallback"
                ) from exc

    manifest = {
        "target": target,
        "counts": {
            "problems": len(problems),
            "chunks": len(chunks),
            "entities": len(entities),
            "relations": len(relations),
        },
        "fallback": fallback,
        "artifacts": sorted(path.name for path in processed_dir.glob("*.json")),
    }
    _write_json(processed_dir / "manifest.json", manifest)
    return manifest


def _load_raw_problems(input_dir: Path) -> tuple[RawProblem, ...]:
    if not input_dir.exists():
        raise IngestionError(f"input directory does not exist: {input_dir}")

    raw_items: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            raw_items.extend(dict(item) for item in payload)
        elif isinstance(payload, dict) and isinstance(payload.get("problems"), list):
            raw_items.extend(dict(item) for item in payload["problems"])
        elif isinstance(payload, dict):
            raw_items.append(payload)
        else:
            raise IngestionError(f"unsupported JSON payload in {path}")

    seen: set[tuple[str, str]] = set()
    problems: list[RawProblem] = []
    for raw in raw_items:
        problem = _clean_problem(RawProblem.from_mapping(raw))
        key = (problem.source.lower(), problem.source_id.lower())
        if key in seen:
            continue
        seen.add(key)
        problems.append(problem)

    if not problems:
        raise IngestionError(f"no raw problems found in {input_dir}")
    return tuple(problems)


def _clean_problem(problem: RawProblem) -> RawProblem:
    return replace(
        problem,
        title=_clean_text(problem.title),
        statement=_clean_text(problem.statement),
        answer=_clean_text(problem.answer),
        solution_hints=tuple(_clean_text(value) for value in problem.solution_hints),
        concepts=tuple(_clean_text(value) for value in problem.concepts if _clean_text(value)),
        tags=tuple(_clean_text(value) for value in problem.tags if _clean_text(value)),
        constraints=tuple(_clean_text(value) for value in problem.constraints if _clean_text(value)),
        editorial=_clean_text(problem.editorial) if problem.editorial is not None else None,
    )


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _build_chunks(problems: tuple[RawProblem, ...]) -> tuple[ProblemChunk, ...]:
    chunks: list[ProblemChunk] = []
    for problem in problems:
        fields = [
            ("statement", problem.statement),
            ("answer", problem.answer),
            *(
                (f"hint-{index}", hint)
                for index, hint in enumerate(problem.solution_hints)
            ),
        ]
        if problem.editorial:
            fields.append(("editorial", problem.editorial))
        for index, (kind, text) in enumerate(fields):
            if not text:
                continue
            chunks.append(
                ProblemChunk(
                    id=f"{problem.id}:{kind}:{index}",
                    problem_id=problem.id,
                    kind="hint" if kind.startswith("hint-") else kind,
                    text=text,
                    index=index,
                    concepts=problem.concepts,
                    metadata={
                        "source": problem.source,
                        "sourceId": problem.source_id,
                        "title": problem.title,
                        "problemType": problem.problem_type,
                    },
                )
            )
    return tuple(chunks)


def _extract_entities_and_relations(
    problems: tuple[RawProblem, ...],
) -> tuple[tuple[EntityRecord, ...], tuple[RelationRecord, ...]]:
    concept_problem_ids: dict[str, set[str]] = defaultdict(set)
    concept_names: dict[str, str] = {}
    pattern_problem_ids: dict[str, set[str]] = defaultdict(set)
    pattern_names: dict[str, str] = {}
    entities: list[EntityRecord] = []
    relations: list[RelationRecord] = []

    for problem in problems:
        entities.append(
            EntityRecord(
                id=problem.id,
                name=problem.title,
                type="problem",
                aliases=(problem.source_id,),
                problem_ids=(problem.id,),
                metadata={
                    "source": problem.source,
                    "sourceId": problem.source_id,
                    "problemType": problem.problem_type,
                },
            )
        )
        pattern_id = f"pattern:{_slug(problem.problem_type)}"
        pattern_names[pattern_id] = problem.problem_type
        pattern_problem_ids[pattern_id].add(problem.id)
        relations.append(
            RelationRecord(
                id=f"{problem.id}->{pattern_id}",
                source_id=problem.id,
                target_id=pattern_id,
                type="HAS_PATTERN",
                evidence=(problem.problem_type,),
            )
        )
        for concept in problem.concepts:
            concept_id = f"concept:{_slug(concept)}"
            concept_names[concept_id] = concept
            concept_problem_ids[concept_id].add(problem.id)
            relations.append(
                RelationRecord(
                    id=f"{problem.id}->{concept_id}",
                    source_id=problem.id,
                    target_id=concept_id,
                    type="REQUIRES",
                    evidence=(concept,),
                )
            )

    for concept_id, name in sorted(concept_names.items()):
        entities.append(
            EntityRecord(
                id=concept_id,
                name=name,
                type=_classify_concept(name),
                problem_ids=tuple(sorted(concept_problem_ids[concept_id])),
                metadata={"origin": "mock-extractor"},
            )
        )
    for pattern_id, name in sorted(pattern_names.items()):
        entities.append(
            EntityRecord(
                id=pattern_id,
                name=name,
                type="pattern",
                problem_ids=tuple(sorted(pattern_problem_ids[pattern_id])),
                metadata={"origin": "mock-extractor"},
            )
        )
    return tuple(entities), tuple(relations)


def _write_bm25_index(path: Path, chunks: tuple[ProblemChunk, ...]) -> None:
    documents = [
        {
            "id": chunk.id,
            "text": chunk.text,
            "problemId": chunk.problem_id,
            "tokens": _tokens(chunk.text),
            "payload": chunk.to_mapping(),
        }
        for chunk in chunks
    ]
    _write_json(path, {"documents": documents})


def _build_qdrant_vectors(
    chunks: tuple[ProblemChunk, ...],
    embedding_provider: EmbeddingProvider,
) -> tuple[tuple[VectorRecord, ...], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    vector_records: list[VectorRecord] = []
    for chunk in chunks:
        vector = tuple(embedding_provider.embed_text(chunk.text))
        payload = chunk.to_mapping()
        records.append(
            {
                "id": chunk.id,
                "vector": list(vector),
                "payload": payload,
            }
        )
        vector_records.append(VectorRecord(id=chunk.id, vector=vector, payload=payload))
    return tuple(vector_records), {"embeddingModel": embedding_provider.model_name, "records": records}


def _write_neo4j_graph(
    path: Path,
    entities: tuple[EntityRecord, ...],
    relations: tuple[RelationRecord, ...],
) -> None:
    _write_json(
        path,
        {
            "entities": [entity.to_mapping() for entity in entities],
            "relations": [relation.to_mapping() for relation in relations],
        },
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "unknown"


def _classify_concept(name: str) -> str:
    lowered = name.lower()
    if lowered in {"bfs", "dfs", "dijkstra", "binary search", "dynamic programming"}:
        return "algorithm"
    if lowered in {"queue", "stack", "heap", "visited array", "array", "hash map"}:
        return "data_structure"
    return "concept"


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())
