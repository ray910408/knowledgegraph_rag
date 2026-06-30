from __future__ import annotations

import json
from typing import Any, Sequence

from ..contracts import EntityRecord, RelationRecord


class Neo4jAdapterError(RuntimeError):
    pass


def _metadata_json(value: Any) -> str:
    return json.dumps(dict(value or {}), ensure_ascii=False, sort_keys=True)


class Neo4jGraphStore:
    def __init__(
        self,
        *,
        driver: Any | None = None,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
    ) -> None:
        if driver is not None:
            self._driver = driver
            return
        try:
            from neo4j import GraphDatabase
        except Exception as exc:  # pragma: no cover - depends on optional install state
            raise Neo4jAdapterError("neo4j driver is not installed") from exc
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def upsert_entities(self, entities: Sequence[EntityRecord]) -> None:
        rows = [
            {
                "id": entity.id,
                "name": entity.name,
                "type": entity.type,
                "aliases": list(entity.aliases),
                "problemIds": list(entity.problem_ids),
                "metadataJson": _metadata_json(entity.metadata),
            }
            for entity in entities
        ]
        if not rows:
            return
        with self._driver.session() as session:
            session.run(
                """
                UNWIND $entities AS entity
                MERGE (n:KnowledgeEntity {id: entity.id})
                SET n.id = entity.id,
                    n.name = entity.name,
                    n.type = entity.type,
                    n.aliases = entity.aliases,
                    n.problemIds = entity.problemIds,
                    n.metadataJson = entity.metadataJson
                """,
                entities=rows,
            )

    def upsert_relations(self, relations: Sequence[RelationRecord]) -> None:
        rows = [
            {
                "id": relation.id,
                "sourceId": relation.source_id,
                "targetId": relation.target_id,
                "type": relation.type,
                "weight": relation.weight,
                "evidence": list(relation.evidence),
                "metadataJson": _metadata_json(relation.metadata),
            }
            for relation in relations
        ]
        if not rows:
            return
        with self._driver.session() as session:
            session.run(
                """
                UNWIND $relations AS relation
                MERGE (source:KnowledgeEntity {id: relation.sourceId})
                MERGE (target:KnowledgeEntity {id: relation.targetId})
                MERGE (source)-[edge:RELATED {id: relation.id}]->(target)
                SET edge.id = relation.id,
                    edge.type = relation.type,
                    edge.weight = relation.weight,
                    edge.evidence = relation.evidence,
                    edge.metadataJson = relation.metadataJson
                """,
                relations=rows,
            )

    def find_paths(
        self,
        source_id: str,
        target_id: str,
        *,
        max_hops: int = 3,
    ) -> tuple[dict[str, object], ...]:
        max_hops = max(1, min(int(max_hops), 6))
        query = f"""
            MATCH path = (source:KnowledgeEntity {{id: $sourceId}})
              -[*1..{max_hops}]->
              (target:KnowledgeEntity {{id: $targetId}})
            RETURN {{
              nodes: [node IN nodes(path) | node.id],
              relations: [edge IN relationships(path) | coalesce(edge.type, type(edge))],
              score: 1.0
            }} AS path
            LIMIT 10
        """
        with self._driver.session() as session:
            result = session.run(query, sourceId=source_id, targetId=target_id)
            return tuple(dict(record["path"]) for record in result)

    def find_related_problem_ids(
        self,
        entity_id: str,
        *,
        top_k: int = 10,
    ) -> tuple[str, ...]:
        top_k = max(1, min(int(top_k), 100))
        metadata_query = """
            MATCH (entity:KnowledgeEntity {id: $entityId})
            UNWIND coalesce(entity.problemIds, []) AS problemId
            RETURN problemId
            LIMIT $topK
        """
        adjacent_problem_query = """
            MATCH (entity:KnowledgeEntity {id: $entityId})-[]-(problem:KnowledgeEntity)
            WHERE problem.type = 'problem'
            RETURN DISTINCT problem.id AS problemId
            ORDER BY problemId
            LIMIT $topK
        """
        params = {"entityId": entity_id, "topK": top_k}
        with self._driver.session() as session:
            problem_ids = _problem_ids_from_records(session.run(metadata_query, **params))
            if problem_ids:
                return problem_ids
            return _problem_ids_from_records(session.run(adjacent_problem_query, **params))


def _problem_ids_from_records(records: Any) -> tuple[str, ...]:
    problem_ids: list[str] = []
    for record in records:
        problem_id: Any = None
        try:
            problem_id = record["problemId"]
        except (KeyError, TypeError):
            try:
                problem_id = record["id"]
            except (KeyError, TypeError):
                pass
        if problem_id is not None:
            problem_ids.append(str(problem_id))
    return tuple(problem_ids)
