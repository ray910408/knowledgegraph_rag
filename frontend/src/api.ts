import type {
  AnalysisRequest,
  AnalysisResponse,
  EvidenceBundle,
  EvidenceEdge,
  EvidenceNode,
  EvidencePath,
  InputKind,
  RequiredConcept,
  RetrievalConfig,
  RetrievalTrace,
  SimilarProblem,
  TraceCandidate
} from "./types";

type UnknownRecord = Record<string, unknown>;

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
  ]
};

const fallbackEvidence: EvidenceBundle = {
  similarProblems: [
    {
      id: "uva-10653",
      title: "Bombs! NO they are Mines!!",
      score: 0.95,
      sharedConcepts: ["BFS", "Queue", "Visited Array"],
      answerHint: "把格子視為無權圖節點，使用 BFS 逐層擴展。"
    }
  ],
  graphPaths: [{ nodes: ["input", "concept:bfs", "uva-10653"], relations: ["MENTIONS", "REQUIRED_BY"] }],
  algorithmEvidence: ["BFS"],
  dataStructureEvidence: ["Queue", "Visited Array"],
  patternEvidence: ["Graph Traversal"],
  commonMistakes: ["忘記在入隊時標記 visited", "Queue 初始化時沒有保留距離"]
};

const fallbackResponse: AnalysisResponse = {
  queryId: "mock-graph-traversal",
  usedMockData: true,
  inputKind: "problem",
  problemType: "圖論遍歷（Graph Traversal）",
  requiredConcepts: [
    {
      id: "bfs",
      name: "BFS",
      kind: "algorithm",
      description: "在無權圖中逐層擴展，用來找最短步數。"
    },
    {
      id: "queue",
      name: "Queue",
      kind: "data_structure",
      description: "維持 BFS 待處理節點順序。"
    },
    {
      id: "visited-array",
      name: "Visited Array",
      kind: "data_structure",
      description: "記錄已入隊或處理過的節點，避免重複擴展。"
    }
  ],
  similarProblems: [
    {
      source: "UVa",
      id: "10653",
      title: "Bombs! NO they are Mines!!",
      reason: "同樣是在障礙網格中尋找最短步數，可用 BFS 建模。",
      sharedConcepts: ["BFS", "Queue", "Visited Array"],
      answerHint: "先定義鄰居，再用 BFS 逐層擴展。"
    },
    {
      source: "LeetCode",
      id: "1091",
      title: "Shortest Path in Binary Matrix",
      reason: "同樣是無權圖最短路徑問題。",
      sharedConcepts: ["BFS", "Queue", "Visited Array"],
      answerHint: "把座標與距離一起放入 Queue。"
    }
  ],
  similarityReason: "都需要在無權圖中找最短步數，因此可以用 BFS 找最短步數。",
  solvingHints: ["先建圖或定義鄰居，再 BFS。", "起點入隊時立刻標記 visited。"],
  commonMistakes: ["忘記標記 visited。", "queue 初始化錯誤，導致距離沒有被正確設定。"],
  evidencePaths: [
    {
      title: "圖論遍歷 BFS 分析證據",
      nodes: [
        { id: "input", label: "輸入內容", type: "problem" },
        { id: "graph-traversal", label: "Graph Traversal", type: "pattern" },
        { id: "bfs", label: "BFS", type: "algorithm" },
        { id: "queue", label: "Queue", type: "data_structure" }
      ],
      edges: [
        { from: "input", to: "graph-traversal", relation: "符合輸入訊號", weight: 1 },
        { from: "graph-traversal", to: "bfs", relation: "需要觀念", weight: 1 },
        { from: "graph-traversal", to: "queue", relation: "需要觀念", weight: 1 }
      ]
    }
  ],
  retrievalConfig: {
    embeddingModel: "BAAI/bge-m3",
    rerankerModel: "BAAI/bge-reranker-v2-m3",
    language: "zh-Hant"
  },
  retrievalTrace: fallbackTrace,
  evidenceBundle: fallbackEvidence,
  contextPreview: [
    "Query Understanding",
    "- intent: problem_search",
    "- keywords: bfs, queue, shortest, path",
    "",
    "Similar Problems",
    "- uva-10653 Bombs! NO they are Mines!!"
  ].join("\n")
};

function asRecord(value: unknown): UnknownRecord | null {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? (value as UnknownRecord) : null;
}

function asString(value: unknown, fallback: string): string {
  return typeof value === "string" && value.trim().length > 0 ? value : fallback;
}

function asNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function asStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
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

function normalizeCandidate(value: unknown): TraceCandidate | null {
  const record = asRecord(value);
  if (!record) {
    return null;
  }
  return {
    id: asString(record.id ?? record.problemId, ""),
    title: asString(record.title, ""),
    source: asString(record.source, ""),
    score: asNumber(record.score, 0),
    concepts: asStringArray(record.concepts),
    problemType: asString(record.problemType ?? record.problem_type, ""),
    payload: asRecord(record.payload) ?? undefined
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
    rerankerScores: candidates("rerankerScores")
  };
}

function normalizeEvidenceBundle(value: unknown): EvidenceBundle {
  const record = asRecord(value) ?? {};
  return {
    similarProblems: pickArray(record, ["similarProblems"]).filter((item): item is UnknownRecord => asRecord(item) !== null),
    graphPaths: pickArray(record, ["graphPaths"]).filter((item): item is UnknownRecord => asRecord(item) !== null),
    algorithmEvidence: asStringArray(record.algorithmEvidence),
    dataStructureEvidence: asStringArray(record.dataStructureEvidence),
    patternEvidence: asStringArray(record.patternEvidence),
    commonMistakes: asStringArray(record.commonMistakes)
  };
}

function normalizeRetrievalConfig(value: unknown): RetrievalConfig {
  const record = asRecord(value) ?? {};
  return {
    embeddingModel: asString(record.embeddingModel ?? record.embedding_model, fallbackResponse.retrievalConfig.embeddingModel),
    rerankerModel: asString(record.rerankerModel ?? record.reranker_model, fallbackResponse.retrievalConfig.rerankerModel),
    language: asString(record.language, fallbackResponse.retrievalConfig.language)
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

  if (requiredConcepts.length === 0 || similarProblems.length === 0) {
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
    retrievalTrace: normalizeTrace(record.retrievalTrace),
    evidenceBundle: normalizeEvidenceBundle(record.evidenceBundle),
    contextPreview: typeof record.contextPreview === "string" ? record.contextPreview : undefined
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
