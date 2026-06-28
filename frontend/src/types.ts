export type RetrievalMode = "hybrid" | "vector" | "graph";

export type InputKind = "problem" | "cpp" | "python" | "unknown";

export interface AnalysisRequest {
  inputText: string;
  mode: RetrievalMode;
  topK: number;
  debug: boolean;
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

export interface MatchedProblem {
  id: string;
  title: string;
  source: string;
  sourceId: string;
  matchKind: string;
  confidence: number;
  score: number;
  sharedConcepts: string[];
  problemType: string;
  answerHint?: string;
  solutionHints?: string[];
  difficulty?: string;
  constraints?: string[];
}

export interface ProviderDescriptor {
  provider: string;
  model: string;
  adapter?: string;
}

export interface CompatibilityWarning {
  adapter: string;
  severity: string;
  message: string;
}

export interface RetrievalConfig {
  embeddingModel: string;
  rerankerModel: string;
  language: string;
  embeddingProvider?: ProviderDescriptor;
  rerankerProvider?: ProviderDescriptor;
}

export interface ScoreMeta {
  stage: string;
  displayLabel: string;
  comparableAcrossStages: boolean;
}

export interface ChunkEvidence {
  available: boolean;
  complete: boolean;
  missingSources: string[];
  unavailableReason: string;
}

export interface CodeFeatures {
  language: string;
  features: string[];
}

export interface TraceCandidate {
  id: string;
  title?: string;
  source?: string;
  candidateSource?: string;
  score?: number;
  concepts?: string[];
  problemType?: string;
  scoreMeta?: ScoreMeta;
  payload?: Record<string, unknown>;
  rawChunks?: TraceCandidate[];
  chunkEvidence?: ChunkEvidence;
}

export interface GraphPathNode {
  id?: string;
  label?: string;
  layer?: string;
}

export interface GraphPathRelation {
  source?: string;
  target?: string;
  type?: string;
  weight?: number;
}

export interface GraphPathScoring {
  strategy?: string;
  score?: number;
  components?: Record<string, number>;
}

export interface GraphPathTrace {
  nodes?: GraphPathNode[];
  relations?: GraphPathRelation[];
  rationale?: string;
  score?: number;
  storePath?: Record<string, unknown>;
  pathSource?: "neo4j" | "inferred" | string;
  graphPathOperation?: "exact_expansion" | "candidate_retrieval" | string;
  pathScoring?: GraphPathScoring;
  scoreMeta?: ScoreMeta;
}

export interface RetrievalTrace {
  queryUnderstanding: {
    originalQuery?: string;
    normalizedQuery?: string;
    inputKind?: InputKind;
    intent?: string;
    keywords?: string[];
    queryLanguage?: "zh-Hant" | "en" | "mixed" | string;
    exactTerms?: string[];
    lowWeightTerms?: string[];
    conceptSeeds?: string[];
    expandedTerms?: string[];
    queryVariants?: {
      bm25?: string;
      vector?: string;
      graphSeeds?: string[];
    };
    codeFeatures?: CodeFeatures;
  };
  entityLinking: Array<Record<string, unknown>>;
  vectorCandidates: TraceCandidate[];
  graphCandidates: TraceCandidate[];
  bm25Candidates: TraceCandidate[];
  fusionScores: TraceCandidate[];
  rerankerScores: TraceCandidate[];
  candidateSources?: Record<string, string>;
  providerSources?: Record<string, ProviderDescriptor>;
  compatibilityWarnings?: CompatibilityWarning[];
  matchedProblem?: MatchedProblem;
}

export interface EvidenceBundle {
  similarProblems: Array<Record<string, unknown>>;
  graphPaths: GraphPathTrace[];
  algorithmEvidence: string[];
  dataStructureEvidence: string[];
  patternEvidence: string[];
  techniqueEvidence?: string[];
  commonMistakes: string[];
  matchedProblem?: MatchedProblem;
}

export type AnalysisStatus = "ok" | "unsupported";

export interface AnalysisResponse {
  queryId: string;
  status: AnalysisStatus;
  abstentionReason?: string;
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
  retrievalTrace?: RetrievalTrace;
  evidenceBundle?: EvidenceBundle;
  contextPreview?: string;
  matchedProblem?: MatchedProblem;
}
