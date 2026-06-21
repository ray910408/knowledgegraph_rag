from __future__ import annotations

from typing import Protocol

from ..domain import EvidenceBundle


class LLMProvider(Protocol):
    def explain(self, evidence_bundle: EvidenceBundle, token: str | None = None) -> str:
        ...


class MockLLMProvider:
    def explain(self, evidence_bundle: EvidenceBundle, token: str | None = None) -> str:
        lines = [
            f"Query: {evidence_bundle.query_problem.title}",
            "Recommendations are limited to retrieved evidence.",
        ]

        if not evidence_bundle.recommendations:
            lines.append("No evidence-backed recommendations were found.")
            return "\n".join(lines)

        for index, recommendation in enumerate(
            evidence_bundle.recommendations,
            start=1,
        ):
            lines.append(
                f"{index}. {recommendation.title} "
                f"(score={recommendation.score}, "
                f"vector={recommendation.vector_score}, "
                f"graph={recommendation.graph_score}, "
                f"concept={recommendation.concept_match_score})"
            )

            if recommendation.concept_matches:
                lines.append(
                    "   concepts: " + ", ".join(recommendation.concept_matches)
                )

            for path in recommendation.evidence_paths:
                lines.append(
                    "   path: "
                    + " -> ".join(path.nodes)
                    + " via "
                    + " -> ".join(path.relations)
                )

        return "\n".join(lines)
