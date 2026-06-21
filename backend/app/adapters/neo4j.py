from __future__ import annotations

from typing import Any, Sequence

from ..contracts import EntityRecord, RelationRecord


class Neo4jAdapterError(RuntimeError):
    pass


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
        rows = [entity.to_mapping() for entity in entities]
        if not rows:
            return
        with self._driver.session() as session:
            session.run(
                """
                UNWIND $entities AS entity
                MERGE (n:KnowledgeEntity {id: entity.id})
                SET n += entity
                """,
                entities=rows,
            )

    def upsert_relations(self, relations: Sequence[RelationRecord]) -> None:
        rows = [relation.to_mapping() for relation in relations]
        if not rows:
            return
        with self._driver.session() as session:
            session.run(
                """
                UNWIND $relations AS relation
                MERGE (source:KnowledgeEntity {id: relation.sourceId})
                MERGE (target:KnowledgeEntity {id: relation.targetId})
                MERGE (source)-[edge:RELATED {id: relation.id}]->(target)
                SET edge.type = relation.type,
                    edge.weight = relation.weight,
                    edge.evidence = relation.evidence,
                    edge.metadata = relation.metadata
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
