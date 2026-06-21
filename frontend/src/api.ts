import type {
  AnalysisRequest,
  AnalysisResponse,
  EvidenceEdge,
  EvidenceNode,
  EvidencePath,
  InputKind,
  RequiredConcept,
  RetrievalConfig,
  SimilarProblem
} from "./types";

type UnknownRecord = Record<string, unknown>;

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
      description: "在無權圖中依照層次展開搜尋，適合找最短步數。"
    },
    {
      id: "queue",
      name: "Queue",
      kind: "data_structure",
      description: "維持 BFS 的先進先出順序，確保同一層先被處理。"
    },
    {
      id: "visited-array",
      name: "Visited Array",
      kind: "data_structure",
      description: "記錄狀態是否已處理，避免重複入列與無限循環。"
    }
  ],
  similarProblems: [
    {
      source: "UVa",
      id: "532",
      title: "Dungeon Master",
      reason: "同樣把狀態視為無權圖節點，使用 BFS 找最短步數。",
      sharedConcepts: ["BFS", "Queue", "Visited Array"],
      answerHint: "把三維座標當狀態，從起點逐層擴展。"
    },
    {
      source: "LeetCode",
      id: "1091",
      title: "Shortest Path in Binary Matrix",
      reason: "同樣需要在無權網格圖中找最短路徑長度。",
      sharedConcepts: ["BFS", "Queue", "Visited Array"],
      answerHint: "八方向鄰居入列時立刻標記 visited。"
    }
  ],
  similarityReason: "都需要在無權圖中用 BFS 找最短步數，並用 Queue 維持搜尋層次。",
  solvingHints: ["先建圖或定義狀態轉移。", "從起點初始化 Queue。", "每次擴展相鄰節點並標記 visited。"],
  commonMistakes: ["忘記標記 visited。", "Queue 初始化錯誤。", "步數層級 off-by-one。"],
  evidencePaths: [
    {
      title: "無權最短路徑推理",
      nodes: [
        { id: "input", label: "輸入題目", type: "problem" },
        { id: "graph-traversal", label: "Graph Traversal", type: "pattern" },
        { id: "bfs", label: "BFS", type: "algorithm" },
        { id: "queue", label: "Queue", type: "data_structure" }
      ],
      edges: [
        { from: "input", to: "graph-traversal", relation: "偵測到圖遍歷訊號", weight: 0.86 },
        { from: "graph-traversal", to: "bfs", relation: "無權最短步數", weight: 0.92 },
        { from: "bfs", to: "queue", relation: "需要資料結構", weight: 0.88 }
      ]
    }
  ],
  retrievalConfig: {
    embeddingModel: "BAAI/bge-m3",
    rerankerModel: "BAAI/bge-reranker-v2-m3",
    language: "zh-Hant"
  }
};

function asRecord(value: unknown): UnknownRecord | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as UnknownRecord)
    : null;
}

function asString(value: unknown, fallback: string): string {
  return typeof value === "string" && value.trim().length > 0 ? value : fallback;
}

function asNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function clamp01(value: unknown, fallback: number): number {
  return Math.min(1, Math.max(0, asNumber(value, fallback)));
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

  if (value === "data-structure" || value === "dataStructure") {
    return "data_structure";
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
    source: asString(record.source, "題庫"),
    id: asString(record.id ?? record.problemId ?? record.problem_id, `${index + 1}`),
    title,
    reason: asString(record.reason ?? record.similarityReason ?? record.similarity_reason, "概念與解法模式相近。"),
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
    relation: asString(record.relation ?? record.label ?? record.type, "關聯"),
    weight: clamp01(record.weight ?? record.score, 0)
  };
}

function normalizeEvidencePath(value: unknown, index: number): EvidencePath | null {
  const record = asRecord(value);
  if (!record) {
    return null;
  }

  const nodes = pickArray(record, ["nodes", "path", "vertices"])
    .map(normalizeEvidenceNode)
    .filter((node): node is EvidenceNode => node !== null);
  const edges = pickArray(record, ["edges", "relations", "links"])
    .map(normalizeEvidenceEdge)
    .filter((edge): edge is EvidenceEdge => edge !== null);

  return {
    title: asString(record.title ?? record.name, `證據路徑 ${index + 1}`),
    nodes,
    edges
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
    usedMockData: true
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
    usedMockData: false,
    inputKind: normalizeInputKind(record.inputKind ?? record.input_kind),
    problemType: asString(record.problemType ?? record.problem_type, "未判定"),
    requiredConcepts,
    similarProblems,
    similarityReason: asString(record.similarityReason ?? record.similarity_reason, "相似題具有共同的解題觀念。"),
    solvingHints: asStringArray(record.solvingHints ?? record.solving_hints ?? record.hints),
    commonMistakes: asStringArray(record.commonMistakes ?? record.common_mistakes ?? record.mistakes),
    evidencePaths: pickArray(record, ["evidencePaths", "evidence_paths", "paths"])
      .map(normalizeEvidencePath)
      .filter((path): path is EvidencePath => path !== null),
    retrievalConfig: normalizeRetrievalConfig(record.retrievalConfig ?? record.retrieval_config)
  };
}

function normalizeRequest(request: AnalysisRequest): UnknownRecord {
  return {
    inputText: request.inputText,
    input_text: request.inputText,
    statement: request.inputText,
    mode: request.mode,
    topK: request.topK,
    top_k: request.topK
  };
}

export async function fetchAnalysis(request: AnalysisRequest): Promise<AnalysisResponse> {
  try {
    const response = await fetch("/api/analysis", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(normalizeRequest(request))
    });

    if (!response.ok) {
      throw new Error(`分析 API 回傳 ${response.status}`);
    }

    return normalizeResponse(await response.json(), request);
  } catch {
    return mockFallback(request);
  }
}
