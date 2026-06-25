import type {
  AnalysisRequest,
  AnalysisResponse,
  EvidenceBundle,
  EvidenceEdge,
  EvidenceNode,
  EvidencePath,
  GraphPathTrace,
  InputKind,
  MatchedProblem,
  ProviderDescriptor,
  RequiredConcept,
  RetrievalConfig,
  RetrievalTrace,
  SimilarProblem,
  TraceCandidate
} from "./types";

type UnknownRecord = Record<string, unknown>;

const MAX_RAW_CHUNK_DEPTH = 4;

const fallbackMatchedProblem: MatchedProblem = {
  id: "uva-10653",
  title: "Bombs! NO they are Mines!!",
  source: "UVa",
  sourceId: "10653",
  matchKind: "exact_problem_id",
  confidence: 1,
  score: 1,
  sharedConcepts: ["BFS", "Queue", "Visited Array"],
  problemType: "Graph Traversal",
  answerHint: "把每個可走格子視為無權圖節點，使用 BFS 找出最短距離。",
  solutionHints: [
    "先定義四方向鄰居，或在 BFS 過程中即時計算。",
    "格子入隊時就標記 visited，避免重複處理。"
  ],
  difficulty: "practice"
};

const fallbackProviderSources: Record<string, ProviderDescriptor> = {
  embedding: {
    provider: "mock",
    model: "BAAI/bge-m3",
    adapter: "local"
  },
  reranker: {
    provider: "mock",
    model: "BAAI/bge-reranker-v2-m3"
  }
};

const fallbackTrace: RetrievalTrace = {
  queryUnderstanding: {
    intent: "problem_search",
    inputKind: "problem",
    keywords: ["bfs", "queue", "shortest", "path"]
  },
  entityLinking: [
    { entityId: "concept:bfs", name: "BFS", type: "algorithm", confidence: 1 },
    { entityId: "concept:queue", name: "Queue", type: "data_structure", confidence: 1 }
  ],
  vectorCandidates: [
    { id: "leetcode-1091", title: "Shortest Path in Binary Matrix", source: "vector", score: 0.86 }
  ],
  graphCandidates: [{ id: "uva-10653", title: "Bombs! NO they are Mines!!", source: "graph", score: 1 }],
  bm25Candidates: [{ id: "leetcode-994", title: "Rotting Oranges", source: "bm25", score: 0.74 }],
  fusionScores: [
    { id: "uva-10653", title: "Bombs! NO they are Mines!!", source: "hybrid", score: 0.92 }
  ],
  rerankerScores: [
    { id: "uva-10653", title: "Bombs! NO they are Mines!!", source: "hybrid", score: 0.95 }
  ],
  providerSources: fallbackProviderSources,
  matchedProblem: fallbackMatchedProblem
};

const fallbackEvidence: EvidenceBundle = {
  similarProblems: [
    {
      id: "leetcode-1091",
      title: "Shortest Path in Binary Matrix",
      score: 0.86,
      sharedConcepts: ["BFS", "Queue", "Visited Array"],
      answerHint: "在格子圖上用 BFS 逐層擴展並記錄距離。"
    },
    {
      id: "leetcode-994",
      title: "Rotting Oranges",
      score: 0.74,
      sharedConcepts: ["BFS", "Queue", "Visited Array"],
      answerHint: "從所有初始腐爛橘子同時開始 BFS。"
    }
  ],
  graphPaths: [
    {
      nodes: ["input", "concept:bfs", "uva-10653"],
      relations: ["MENTIONS", "REQUIRED_BY"],
      rationale: "前端 mock fallback 推論出的示意路徑，未來自 Neo4j。",
      pathSource: "inferred"
    }
  ],
  algorithmEvidence: ["BFS"],
  dataStructureEvidence: ["Queue"],
  patternEvidence: ["Graph Traversal"],
  techniqueEvidence: ["Visited Array"],
  commonMistakes: ["入隊時忘記標記 visited", "處理 Queue 時沒有保留距離層數"],
  matchedProblem: fallbackMatchedProblem
};

const fallbackResponse: AnalysisResponse = {
  queryId: "mock-graph-traversal",
  usedMockData: true,
  inputKind: "problem",
  problemType: "Graph Traversal",
  requiredConcepts: [
    {
      id: "bfs",
      name: "BFS",
      kind: "algorithm",
      description: "在無權圖中逐層擴展，用來找最短距離。"
    },
    {
      id: "queue",
      name: "Queue",
      kind: "data_structure",
      description: "以先進先出的順序維護 BFS 待處理節點。"
    },
    {
      id: "visited-array",
      name: "Visited Array",
      kind: "data_structure",
      description: "記錄已入隊或已處理的格子，避免重複擴展。"
    }
  ],
  similarProblems: [
    {
      source: "LeetCode",
      id: "1091",
      title: "Shortest Path in Binary Matrix",
      reason: "同樣是在無權格子圖中尋找最短路徑。",
      sharedConcepts: ["BFS", "Queue", "Visited Array"],
      answerHint: "把座標與距離一起放入 Queue。"
    },
    {
      source: "LeetCode",
      id: "994",
      title: "Rotting Oranges",
      reason: "同樣使用 Queue 在格子圖上逐層擴展 BFS。",
      sharedConcepts: ["BFS", "Queue", "Visited Array"],
      answerHint: "先把所有初始起點放入 Queue。"
    }
  ],
  similarityReason: "精確匹配的 UVa 題目會獨立顯示；相似題只保留可練習同類 BFS 技巧的題目。",
  solvingHints: ["先定義合法鄰居，再執行 BFS。", "節點入隊時就標記 visited。"],
  commonMistakes: ["忘記標記 visited", "沒有依 BFS 層數維護距離"],
  evidencePaths: [
    {
      title: "Graph Traversal BFS 分析證據",
      nodes: [
        { id: "input", label: "輸入查詢", type: "problem" },
        { id: "graph-traversal", label: "Graph Traversal", type: "pattern" },
        { id: "bfs", label: "BFS", type: "algorithm" },
        { id: "queue", label: "Queue", type: "data_structure" }
      ],
      edges: [
        { from: "input", to: "graph-traversal", relation: "符合輸入訊號", weight: 1 },
        { from: "graph-traversal", to: "bfs", relation: "需要觀念", weight: 1 },
        { from: "graph-traversal", to: "queue", relation: "使用資料結構", weight: 1 }
      ]
    }
  ],
  retrievalConfig: {
    embeddingModel: "BAAI/bge-m3",
    rerankerModel: "BAAI/bge-reranker-v2-m3",
    language: "zh-Hant",
    embeddingProvider: fallbackProviderSources.embedding,
    rerankerProvider: fallbackProviderSources.reranker
  },
  retrievalTrace: fallbackTrace,
  evidenceBundle: fallbackEvidence,
  contextPreview: [
    "Query Understanding",
    "- intent: problem_search",
    "- keywords: bfs, queue, shortest, path",
    "",
    "Matched Problem",
    "- uva-10653 Bombs! NO they are Mines!!",
    "",
    "Similar Problems",
    "- leetcode-1091 Shortest Path in Binary Matrix",
    "- leetcode-994 Rotting Oranges"
  ].join("\n"),
  matchedProblem: fallbackMatchedProblem
};

function asRecord(value: unknown): UnknownRecord | null {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? (value as UnknownRecord) : null;
}

function asString(value: unknown, fallback: string): string {
  return typeof value === "string" && value.trim().length > 0 ? value : fallback;
}

function asOptionalString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim().length > 0 ? value : undefined;
}

function asNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function asStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function asStringRecord(value: unknown): Record<string, string> | undefined {
  const record = asRecord(value);
  if (!record) {
    return undefined;
  }
  const entries = Object.entries(record)
    .filter((entry): entry is [string, string] => typeof entry[1] === "string")
    .sort(([left], [right]) => left.localeCompare(right));
  return entries.length > 0 ? Object.fromEntries(entries) : undefined;
}

function firstPresent(record: UnknownRecord, keys: string[]): unknown {
  for (const key of keys) {
    if (key in record) {
      return record[key];
    }
  }
  return undefined;
}

function pickArray(record: UnknownRecord, keys: string[]): unknown[] {
  for (const key of keys) {
    const value = record[key];
    if (Array.isArray(value)) {
      return value;
    }
  }
  return [];
}

function normalizeInputKind(value: unknown): InputKind {
  return value === "problem" || value === "cpp" || value === "python" ? value : "unknown";
}

function normalizeConceptKind(value: unknown): RequiredConcept["kind"] {
  if (value === "algorithm" || value === "data_structure" || value === "pattern" || value === "concept") {
    return value;
  }
  return "concept";
}

function normalizeNodeType(value: unknown): EvidenceNode["type"] {
  if (
    value === "problem" ||
    value === "concept" ||
    value === "algorithm" ||
    value === "data_structure" ||
    value === "pattern"
  ) {
    return value;
  }
  return "concept";
}

function normalizeRequiredConcept(value: unknown, index: number): RequiredConcept | null {
  const record = asRecord(value);
  if (!record) {
    return null;
  }
  const name = asString(record.name ?? record.title ?? record.label, "");
  if (!name) {
    return null;
  }
  return {
    id: asString(record.id ?? record.key, `concept-${index + 1}`),
    name,
    kind: normalizeConceptKind(record.kind ?? record.type),
    description: asString(record.description ?? record.summary, "")
  };
}

function normalizeSimilarProblem(value: unknown, index: number): SimilarProblem | null {
  const record = asRecord(value);
  if (!record) {
    return null;
  }
  const title = asString(record.title ?? record.name, "");
  if (!title) {
    return null;
  }
  return {
    source: asString(record.source, "unknown"),
    id: asString(record.id ?? record.problemId ?? record.problem_id, `${index + 1}`),
    title,
    reason: asString(record.reason ?? record.similarityReason ?? record.similarity_reason, ""),
    sharedConcepts: asStringArray(record.sharedConcepts ?? record.shared_concepts ?? record.concepts),
    answerHint: asString(record.answerHint ?? record.answer_hint ?? record.solutionHint ?? record.solution_hint, "")
  };
}

function normalizeEvidenceNode(value: unknown, index: number): EvidenceNode | null {
  const record = asRecord(value);
  if (!record) {
    return null;
  }
  const label = asString(record.label ?? record.name ?? record.title, "");
  if (!label) {
    return null;
  }
  return {
    id: asString(record.id ?? record.key, `node-${index + 1}`),
    label,
    type: normalizeNodeType(record.type ?? record.kind)
  };
}

function normalizeEvidenceEdge(value: unknown, index: number): EvidenceEdge | null {
  const record = asRecord(value);
  if (!record) {
    return null;
  }
  return {
    from: asString(record.from ?? record.source, `node-${index + 1}`),
    to: asString(record.to ?? record.target, `node-${index + 2}`),
    relation: asString(record.relation ?? record.label ?? record.type, "RELATED"),
    weight: Math.min(1, Math.max(0, asNumber(record.weight ?? record.score, 0)))
  };
}

function normalizeEvidencePath(value: unknown, index: number): EvidencePath | null {
  const record = asRecord(value);
  if (!record) {
    return null;
  }
  return {
    title: asString(record.title ?? record.name, `證據路徑 ${index + 1}`),
    nodes: pickArray(record, ["nodes", "path", "vertices"])
      .map(normalizeEvidenceNode)
      .filter((node): node is EvidenceNode => node !== null),
    edges: pickArray(record, ["edges", "relations", "links"])
      .map(normalizeEvidenceEdge)
      .filter((edge): edge is EvidenceEdge => edge !== null)
  };
}

function normalizeMatchedProblem(value: unknown): MatchedProblem | undefined {
  const record = asRecord(value);
  if (!record) {
    return undefined;
  }
  const id = asString(record.id ?? record.problemId ?? record.problem_id, "");
  if (!id) {
    return undefined;
  }
  return {
    id,
    title: asString(record.title ?? record.name, id),
    source: asString(record.source, ""),
    sourceId: asString(record.sourceId ?? record.source_id, ""),
    matchKind: asString(record.matchKind ?? record.match_kind, ""),
    confidence: asNumber(record.confidence, 0),
    score: asNumber(record.score, 0),
    sharedConcepts: asStringArray(record.sharedConcepts ?? record.shared_concepts ?? record.concepts),
    problemType: asString(record.problemType ?? record.problem_type, ""),
    answerHint: asOptionalString(record.answerHint ?? record.answer_hint ?? record.answer),
    solutionHints: asStringArray(record.solutionHints ?? record.solution_hints),
    difficulty: asOptionalString(record.difficulty),
    constraints: asStringArray(record.constraints)
  };
}

function normalizeProviderDescriptor(value: unknown): ProviderDescriptor | undefined {
  const record = asRecord(value);
  if (!record) {
    return undefined;
  }
  const provider = asString(record.provider, "");
  const model = asString(record.model, "");
  if (!provider || !model) {
    return undefined;
  }
  return {
    provider,
    model,
    adapter: asOptionalString(record.adapter)
  };
}

function normalizeProviderSources(value: unknown): Record<string, ProviderDescriptor> | undefined {
  const record = asRecord(value);
  if (!record) {
    return undefined;
  }
  const entries = Object.entries(record)
    .map(([key, descriptor]) => [key, normalizeProviderDescriptor(descriptor)] as const)
    .filter((entry): entry is readonly [string, ProviderDescriptor] => entry[1] !== undefined)
    .sort(([left], [right]) => left.localeCompare(right));
  return entries.length > 0 ? Object.fromEntries(entries) : undefined;
}

function normalizeGraphPath(value: unknown): GraphPathTrace | null {
  const record = asRecord(value);
  if (!record) {
    return null;
  }
  return {
    nodes: Array.isArray(record.nodes) ? record.nodes : [],
    relations: Array.isArray(record.relations) ? record.relations : [],
    rationale: asString(record.rationale, ""),
    score: typeof record.score === "number" && Number.isFinite(record.score) ? record.score : undefined,
    storePath: asRecord(record.storePath ?? record.store_path) ?? undefined,
    pathSource: asOptionalString(record.pathSource ?? record.path_source)
  };
}

function normalizeCandidate(value: unknown, depth = 0): TraceCandidate | null {
  const record = asRecord(value);
  if (!record) {
    return null;
  }
  const payload = asRecord(record.payload);
  const rawChunks = payload && depth < MAX_RAW_CHUNK_DEPTH
    ? pickArray(payload, ["rawChunks", "raw_chunks"])
        .map((candidate) => normalizeCandidate(candidate, depth + 1))
        .filter((candidate): candidate is TraceCandidate => candidate !== null && candidate.id.length > 0)
    : [];
  return {
    id: asString(record.id ?? record.problemId ?? record.problem_id, ""),
    title: asString(record.title, ""),
    source: asString(record.source, ""),
    candidateSource: asString(record.candidateSource ?? record.candidate_source, ""),
    score: asNumber(record.score, 0),
    concepts: asStringArray(record.concepts),
    problemType: asString(record.problemType ?? record.problem_type, ""),
    payload: payload ?? undefined,
    rawChunks: rawChunks.length > 0 ? rawChunks : undefined
  };
}

function normalizeTrace(value: unknown): RetrievalTrace {
  const record = asRecord(value) ?? {};
  const queryUnderstanding = asRecord(record.queryUnderstanding) ?? fallbackTrace.queryUnderstanding;
  const candidates = (key: string) =>
    pickArray(record, [key])
      .map(normalizeCandidate)
      .filter((candidate): candidate is TraceCandidate => candidate !== null && candidate.id.length > 0);
  return {
    queryUnderstanding: {
      originalQuery: asString(queryUnderstanding.originalQuery, ""),
      normalizedQuery: asString(queryUnderstanding.normalizedQuery, ""),
      inputKind: normalizeInputKind(queryUnderstanding.inputKind),
      intent: asString(queryUnderstanding.intent, "problem_search"),
      keywords: asStringArray(queryUnderstanding.keywords)
    },
    entityLinking: pickArray(record, ["entityLinking"]).filter((item): item is UnknownRecord => asRecord(item) !== null),
    vectorCandidates: candidates("vectorCandidates"),
    graphCandidates: candidates("graphCandidates"),
    bm25Candidates: candidates("bm25Candidates"),
    fusionScores: candidates("fusionScores"),
    rerankerScores: candidates("rerankerScores"),
    candidateSources: asStringRecord(firstPresent(record, ["candidateSources", "candidate_sources"])),
    providerSources: normalizeProviderSources(firstPresent(record, ["providerSources", "provider_sources"])),
    matchedProblem: normalizeMatchedProblem(firstPresent(record, ["matchedProblem", "matched_problem"]))
  };
}

function normalizeEvidenceBundle(value: unknown): EvidenceBundle {
  const record = asRecord(value) ?? {};
  return {
    similarProblems: pickArray(record, ["similarProblems"]).filter((item): item is UnknownRecord => asRecord(item) !== null),
    graphPaths: pickArray(record, ["graphPaths", "graph_paths"])
      .map(normalizeGraphPath)
      .filter((item): item is GraphPathTrace => item !== null),
    algorithmEvidence: asStringArray(record.algorithmEvidence ?? record.algorithm_evidence),
    dataStructureEvidence: asStringArray(record.dataStructureEvidence ?? record.data_structure_evidence),
    patternEvidence: asStringArray(record.patternEvidence ?? record.pattern_evidence),
    techniqueEvidence: asStringArray(record.techniqueEvidence ?? record.technique_evidence),
    commonMistakes: asStringArray(record.commonMistakes ?? record.common_mistakes),
    matchedProblem: normalizeMatchedProblem(firstPresent(record, ["matchedProblem", "matched_problem"]))
  };
}

function normalizeRetrievalConfig(value: unknown): RetrievalConfig {
  const record = asRecord(value) ?? {};
  return {
    embeddingModel: asString(record.embeddingModel ?? record.embedding_model, fallbackResponse.retrievalConfig.embeddingModel),
    rerankerModel: asString(record.rerankerModel ?? record.reranker_model, fallbackResponse.retrievalConfig.rerankerModel),
    language: asString(record.language, fallbackResponse.retrievalConfig.language),
    embeddingProvider: normalizeProviderDescriptor(
      firstPresent(record, ["embeddingProvider", "embedding_provider"])
    ),
    rerankerProvider: normalizeProviderDescriptor(firstPresent(record, ["rerankerProvider", "reranker_provider"]))
  };
}

function mockFallback(request: AnalysisRequest): AnalysisResponse {
  return {
    ...fallbackResponse,
    queryId: `mock-${request.mode}-${request.topK}`,
    similarProblems: fallbackResponse.similarProblems.slice(0, request.topK),
    usedMockData: true,
    contextPreview: request.debug ? fallbackResponse.contextPreview : undefined
  };
}

function normalizeResponse(payload: unknown, request: AnalysisRequest): AnalysisResponse {
  const record = asRecord(payload);
  if (!record) {
    return mockFallback(request);
  }

  const requiredConcepts = pickArray(record, ["requiredConcepts", "required_concepts", "concepts"])
    .map(normalizeRequiredConcept)
    .filter((item): item is RequiredConcept => item !== null);
  const similarProblems = pickArray(record, ["similarProblems", "similar_problems", "recommendations"])
    .map(normalizeSimilarProblem)
    .filter((item): item is SimilarProblem => item !== null)
    .slice(0, request.topK);
  const matchedProblem = normalizeMatchedProblem(firstPresent(record, ["matchedProblem", "matched_problem"]));

  if (requiredConcepts.length === 0 || (similarProblems.length === 0 && !matchedProblem)) {
    return mockFallback(request);
  }

  return {
    queryId: asString(record.queryId ?? record.query_id ?? record.id, `api-${Date.now()}`),
    usedMockData: Boolean(record.usedMockData ?? record.used_mock_data ?? false),
    inputKind: normalizeInputKind(record.inputKind ?? record.input_kind),
    problemType: asString(record.problemType ?? record.problem_type, "未知題型"),
    requiredConcepts,
    similarProblems,
    similarityReason: asString(record.similarityReason ?? record.similarity_reason, ""),
    solvingHints: asStringArray(record.solvingHints ?? record.solving_hints ?? record.hints),
    commonMistakes: asStringArray(record.commonMistakes ?? record.common_mistakes ?? record.mistakes),
    evidencePaths: pickArray(record, ["evidencePaths", "evidence_paths", "paths"])
      .map(normalizeEvidencePath)
      .filter((path): path is EvidencePath => path !== null),
    retrievalConfig: normalizeRetrievalConfig(record.retrievalConfig ?? record.retrieval_config),
    retrievalTrace: normalizeTrace(firstPresent(record, ["retrievalTrace", "retrieval_trace"])),
    evidenceBundle: normalizeEvidenceBundle(firstPresent(record, ["evidenceBundle", "evidence_bundle"])),
    contextPreview: asOptionalString(firstPresent(record, ["contextPreview", "context_preview"])),
    matchedProblem
  };
}

export async function fetchAnalysis(request: AnalysisRequest): Promise<AnalysisResponse> {
  try {
    const response = await fetch(`/api/analysis${request.debug ? "?debug=true" : ""}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        input: request.inputText,
        mode: request.mode,
        topK: request.topK
      })
    });

    if (!response.ok) {
      throw new Error(`analysis API returned ${response.status}`);
    }

    return normalizeResponse(await response.json(), request);
  } catch {
    return mockFallback(request);
  }
}
