export type RetrievalMode = "hybrid" | "vector" | "graph";

export type InputKind = "problem" | "cpp" | "python" | "unknown";

export interface AnalysisRequest {
  inputText: string;
  mode: RetrievalMode;
  topK: number;
}

export interface RequiredConcept {
  id: string;
  name: string;
  kind: "algorithm" | "data_structure" | "pattern" | "concept";
  description: string;
}

export interface SimilarProblem {
  source: string;
  id: string;
  title: string;
  reason: string;
  sharedConcepts: string[];
  answerHint?: string;
}

export interface EvidenceNode {
  id: string;
  label: string;
  type: "problem" | "concept" | "algorithm" | "data_structure" | "pattern";
}

export interface EvidenceEdge {
  from: string;
  to: string;
  relation: string;
  weight: number;
}

export interface EvidencePath {
  title: string;
  nodes: EvidenceNode[];
  edges: EvidenceEdge[];
}

export interface RetrievalConfig {
  embeddingModel: string;
  rerankerModel: string;
  language: string;
}

export interface AnalysisResponse {
  queryId: string;
  usedMockData: boolean;
  inputKind: InputKind;
  problemType: string;
  requiredConcepts: RequiredConcept[];
  similarProblems: SimilarProblem[];
  similarityReason: string;
  solvingHints: string[];
  commonMistakes: string[];
  evidencePaths: EvidencePath[];
  retrievalConfig: RetrievalConfig;
}
