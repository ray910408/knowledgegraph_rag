import { useRef, useState } from "react";
import type { ReactNode } from "react";
import { fetchAnalysis } from "./api";
import type {
  AnalysisResponse,
  EvidenceBundle,
  GraphPathTrace,
  EvidencePath,
  InputKind,
  MatchedProblem,
  RequiredConcept,
  RetrievalMode,
  RetrievalTrace,
  SimilarProblem,
  TraceCandidate
} from "./types";

const sampleInput =
  "給定一張無權圖與起點、終點，請找出從起點到終點的最短步數。需要說明該使用哪些演算法與資料結構。";
const emptyInputError = "請輸入題目、題號、關鍵字或程式碼。";

const maxAnalysisInputChars = 8000;

const modes: Array<{ id: RetrievalMode; label: string }> = [
  { id: "hybrid", label: "Hybrid" },
  { id: "vector", label: "Vector" },
  { id: "graph", label: "Graph" }
];

const flowSteps = ["輸入", "查詢理解", "三路檢索", "Fusion / Rerank", "Evidence / Context", "回答"];

function clampTopK(value: number, fallback: number): number {
  if (!Number.isFinite(value)) {
    return fallback;
  }
  return Math.min(8, Math.max(1, Math.trunc(value)));
}

function inputKindLabel(kind: InputKind): string {
  if (kind === "cpp") {
    return "C++ 程式碼";
  }
  if (kind === "python") {
    return "Python 程式碼";
  }
  if (kind === "problem") {
    return "題目敘述";
  }
  return "一般查詢";
}

function conceptKindLabel(kind: RequiredConcept["kind"]): string {
  if (kind === "algorithm") {
    return "演算法";
  }
  if (kind === "data_structure") {
    return "資料結構";
  }
  if (kind === "pattern") {
    return "題型模式";
  }
  return "概念";
}

function formatScore(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "-";
}

function formatLabeledScore(label: string | undefined, value: unknown): string {
  const score = formatScore(value);
  return label ? `${label}: ${score}` : score;
}

function asText(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function candidateKey(title: string, candidate: TraceCandidate, index: number): string {
  const storeCandidateId = asText(candidate.payload?.storeCandidateId ?? candidate.payload?.store_candidate_id);
  const rawChunkIds = candidate.rawChunks
    ?.map((chunk) => asText(chunk.payload?.storeCandidateId ?? chunk.payload?.store_candidate_id))
    .filter(Boolean)
    .join("|");
  return `${title}-${candidate.title ?? ""}-${candidate.id}-${storeCandidateId || rawChunkIds || index}`;
}

function chunkEvidenceBadge(candidate: TraceCandidate): string | null {
  const evidence = candidate.chunkEvidence;
  if (!evidence) {
    return null;
  }
  if (evidence.complete) {
    return "chunks complete";
  }
  if (evidence.missingSources.length > 0) {
    return `missing ${evidence.missingSources.join(", ")}`;
  }
  return evidence.unavailableReason || (evidence.available ? "chunks incomplete" : "chunks unavailable");
}

function JsonBlock({ value }: { value: unknown }) {
  return <pre className="json-block">{JSON.stringify(value ?? null, null, 2)}</pre>;
}

function formatPathItem(value: unknown): string {
  if (value === null || value === undefined) {
    return "";
  }
  if (typeof value !== "object") {
    return String(value);
  }

  if (!Array.isArray(value)) {
    const record = value as Record<string, unknown>;
    const labelParts = ["label", "name", "id", "type"]
      .map((key) => record[key])
      .filter((item) => item !== null && item !== undefined && typeof item !== "object")
      .map((item) => String(item))
      .filter(Boolean);

    if (labelParts.length > 0) {
      return labelParts.join(" / ");
    }
  }

  try {
    const serialized = JSON.stringify(value);
    return serialized || "";
  } catch {
    return Array.isArray(value) ? "[array]" : "{object}";
  }
}

function formatPathPart(value: unknown): string {
  if (!Array.isArray(value)) {
    return "-";
  }
  return value.map(formatPathItem).filter(Boolean).join(" -> ") || "-";
}

function formatPathScoringComponents(components?: Record<string, number>): string {
  if (!components) {
    return "-";
  }
  const entries = Object.entries(components).filter(([, value]) => Number.isFinite(value));
  if (entries.length === 0) {
    return "-";
  }
  return entries.map(([key, value]) => `${key}=${formatScore(value)}`).join(", ");
}

function graphPathSourceLabel(source?: string): string {
  if (source === "inferred") {
    return "推論 fallback";
  }
  return source || "-";
}

function graphPathOperationLabel(operation?: string): string {
  if (operation === "exact_expansion") {
    return "精確展開";
  }
  if (operation === "candidate_retrieval") {
    return "候選檢索";
  }
  return operation || "-";
}

export default function App() {
  const [inputText, setInputText] = useState(sampleInput);
  const [mode, setMode] = useState<RetrievalMode>("hybrid");
  const [topK, setTopK] = useState(4);
  const [debug, setDebug] = useState(true);
  const [response, setResponse] = useState<AnalysisResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const latestRequestId = useRef(0);
  const showResponseDetails = response?.status !== "unsupported";

  async function handleAnalyze() {
    const rawInput = inputText;
    const trimmedInput = rawInput.trim();
    const requestId = latestRequestId.current + 1;
    latestRequestId.current = requestId;

    if (trimmedInput.length === 0 && rawInput.length <= maxAnalysisInputChars) {
      setIsLoading(false);
      setResponse(null);
      setError(emptyInputError);
      return;
    }

    setIsLoading(true);
    setError(null);
    setResponse(null);

    try {
      const nextResponse = await fetchAnalysis({ inputText: rawInput, mode, topK, debug });
      if (requestId !== latestRequestId.current) {
        return;
      }
      setResponse(nextResponse);
    } catch (caughtError) {
      if (requestId !== latestRequestId.current) {
        return;
      }
      setResponse(null);
      setError(caughtError instanceof Error ? caughtError.message : "分析失敗，請稍後再試。");
    } finally {
      if (requestId === latestRequestId.current) {
        setIsLoading(false);
      }
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Knowledge Graph + Hybrid RAG</p>
          <h1>程式題庫查詢工作台</h1>
        </div>
        <div className="system-status" aria-live="polite">
          <span className={response?.usedMockData ? "status-dot mock" : "status-dot"} />
          {response?.usedMockData ? "Mock fallback" : "API ready"}
        </div>
      </header>

      <section className="workspace-grid">
        <aside className="panel input-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">輸入</p>
              <h2>題目 / 題號 / 關鍵字 / 程式碼</h2>
            </div>
            <button className="ghost-button" type="button" onClick={() => setInputText(sampleInput)}>
              載入範例
            </button>
          </div>

          <textarea
            aria-describedby={error ? "analysis-error" : undefined}
            aria-invalid={Boolean(error)}
            aria-label="題目、題號、關鍵字或程式碼"
            value={inputText}
            onChange={(event) => setInputText(event.target.value)}
            placeholder="輸入題目敘述、LeetCode / CPE 題號、關鍵字，或貼上 C++ / Python 程式碼。"
            spellCheck={false}
          />

          <div className="control-group">
            <label>檢索模式</label>
            <div className="segmented-control" role="group" aria-label="檢索模式">
              {modes.map((item) => (
                <button
                  key={item.id}
                  aria-pressed={mode === item.id}
                  className={mode === item.id ? "active" : ""}
                  type="button"
                  onClick={() => setMode(item.id)}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>

          <div className="control-row">
            <label htmlFor="top-k">候選數</label>
            <input
              id="top-k"
              min="1"
              max="8"
              type="number"
              value={topK}
              onChange={(event) => setTopK((current) => clampTopK(Number(event.target.value), current))}
            />
          </div>

          <label className="check-row">
            <input checked={debug} type="checkbox" onChange={(event) => setDebug(event.target.checked)} />
            顯示 context preview
          </label>

          <button className="primary-button" type="button" onClick={handleAnalyze} disabled={isLoading}>
            {isLoading ? "分析中..." : "執行查詢"}
          </button>

          {error && (
            <p className="error-text" id="analysis-error">
              {error}
            </p>
          )}
        </aside>

        <section className="panel result-panel" aria-live="polite">
          {response ? <AnalysisResult response={response} /> : <EmptyState />}
        </section>

        <section className="side-stack">
          <ModelPanel response={response} />
          {showResponseDetails && (
            <>
              <EvidencePathPanel paths={response?.evidencePaths ?? []} />
              <GraphPathsPanel paths={response?.evidenceBundle?.graphPaths ?? []} />
              <ContextPanel contextPreview={response?.contextPreview} />
            </>
          )}
        </section>
      </section>
    </main>
  );
}

function EmptyState() {
  return (
    <div className="empty-state">
      <h2>等待查詢</h2>
      <p>結果會依照線上查詢流程呈現。</p>
    </div>
  );
}

function AnalysisResult({ response }: { response: AnalysisResponse }) {
  if (response.status === "unsupported") {
    return (
      <div className="analysis-result">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">分析結果</p>
            <h2>暫不支援此輸入</h2>
          </div>
          <span className="query-id">{response.queryId}</span>
        </div>

        <OutputBlock title="系統已停止一般檢索流程">
          <p>
            {response.abstentionReason?.trim() ||
              "這次輸入不屬於目前支援的程式題目分析範圍，未產生題目比對或圖檢索證據。"}
          </p>
        </OutputBlock>
      </div>
    );
  }

  const trace = response.retrievalTrace;
  const evidence = response.evidenceBundle;
  const matchedProblem = response.matchedProblem ?? evidence?.matchedProblem ?? trace?.matchedProblem;

  return (
    <div className="analysis-result">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">線上查詢流程</p>
          <h2>{inputKindLabel(response.inputKind)}</h2>
        </div>
        <span className="query-id">{response.queryId}</span>
      </div>

      <FlowStrip />

      <OutputBlock title="回答">
        <p className="problem-type">{response.problemType}</p>
        <p>{response.similarityReason}</p>
      </OutputBlock>

      <MatchedProblemPanel problem={matchedProblem} />
      <QueryUnderstandingPanel trace={trace} />
      <RetrievalPanel trace={trace} />
      <FusionPanel trace={trace} />
      <TraceJsonPanel trace={trace} />
      <EvidenceBundlePanel evidence={evidence} />

      <OutputBlock title="必要概念">
        <div className="concept-grid">
          {response.requiredConcepts.map((concept) => (
            <article className="concept-card" key={concept.id}>
              <div>
                <h3>{concept.name}</h3>
                <span>{conceptKindLabel(concept.kind)}</span>
              </div>
              <p>{concept.description}</p>
            </article>
          ))}
        </div>
      </OutputBlock>

      {response.similarProblems.length > 0 && (
        <OutputBlock title="相似題">
          <div className="similar-list">
            {response.similarProblems.map((problem) => (
              <SimilarProblemCard key={`${problem.source}-${problem.id}`} problem={problem} />
            ))}
          </div>
        </OutputBlock>
      )}

      <TwoColumnLists
        leftTitle="解題提示"
        leftItems={response.solvingHints}
        rightTitle="常見錯誤"
        rightItems={response.commonMistakes}
      />
    </div>
  );
}

function MatchedProblemPanel({ problem }: { problem?: MatchedProblem }) {
  if (!problem) {
    return null;
  }

  return (
    <OutputBlock title="命中題目">
      <p className="problem-type">
        {problem.sourceId || problem.id} - {problem.title}
      </p>
      <div className="kv-grid">
        <div>
          <span>ID</span>
          <strong>{problem.id}</strong>
        </div>
        <div>
          <span>來源</span>
          <strong>{problem.source || "-"}</strong>
        </div>
        <div>
          <span>命中方式</span>
          <strong>{problem.matchKind || "-"}</strong>
        </div>
        <div>
          <span>信心</span>
          <strong>{formatScore(problem.confidence)}</strong>
        </div>
      </div>
      {problem.sharedConcepts.length > 0 && (
        <div className="chips">
          {problem.sharedConcepts.map((concept, index) => (
            <span key={`${concept}-${index}`}>{concept}</span>
          ))}
        </div>
      )}
      {problem.answerHint && <p>{problem.answerHint}</p>}
    </OutputBlock>
  );
}

function FlowStrip() {
  return (
    <ol className="flow-strip">
      {flowSteps.map((step) => (
        <li key={step}>{step}</li>
      ))}
    </ol>
  );
}

function QueryUnderstandingPanel({ trace }: { trace?: RetrievalTrace }) {
  const understanding = trace?.queryUnderstanding;
  const entities = trace?.entityLinking ?? [];
  const codeFeatures = understanding?.codeFeatures?.features ?? [];

  return (
    <OutputBlock title="查詢理解">
      <div className="kv-grid">
        <div>
          <span>意圖</span>
          <strong>{understanding?.intent ?? "-"}</strong>
        </div>
        <div>
          <span>輸入類型</span>
          <strong>{understanding?.inputKind ?? "-"}</strong>
        </div>
        <div>
          <span>關鍵詞</span>
          <strong>{understanding?.keywords?.join(", ") || "-"}</strong>
        </div>
        {codeFeatures.length > 0 && (
          <div>
            <span>程式特徵</span>
            <strong>{codeFeatures.map((feature) => feature.replace(/_/g, " ")).join(", ")}</strong>
          </div>
        )}
      </div>
      <div className="chips">
        {entities.map((entity, index) => (
          <span key={`${asText(entity.entityId)}-${index}`}>
            {asText(entity.name) || asText(entity.entityId) || "entity"}
          </span>
        ))}
      </div>
    </OutputBlock>
  );
}

function RetrievalPanel({ trace }: { trace?: RetrievalTrace }) {
  return (
    <OutputBlock title="三路檢索">
      <div className="retrieval-grid">
        <CandidateList title="向量搜尋 / Qdrant" candidates={trace?.vectorCandidates ?? []} />
        <CandidateList title="圖搜尋 / Neo4j" candidates={trace?.graphCandidates ?? []} />
        <CandidateList title="BM25 關鍵字搜尋" candidates={trace?.bm25Candidates ?? []} />
      </div>
    </OutputBlock>
  );
}

function FusionPanel({ trace }: { trace?: RetrievalTrace }) {
  return (
    <OutputBlock title="混合融合 / 重排序">
      <div className="retrieval-grid two">
        <CandidateList title="融合分數" candidates={trace?.fusionScores ?? []} />
        <CandidateList title="重排序分數" candidates={trace?.rerankerScores ?? []} />
      </div>
    </OutputBlock>
  );
}

function TraceJsonPanel({ trace }: { trace?: RetrievalTrace }) {
  if (!trace) {
    return null;
  }

  return (
    <OutputBlock title="Retrieval Trace">
      <details className="trace-json-details">
        <summary>JSON</summary>
        <JsonBlock value={trace} />
      </details>
    </OutputBlock>
  );
}

function EvidenceBundlePanel({ evidence }: { evidence?: EvidenceBundle }) {
  if (!evidence) {
    return null;
  }

  return (
    <OutputBlock title="證據整理">
      <div className="evidence-bundle">
        <EvidenceList title="演算法" items={evidence.algorithmEvidence} />
        <EvidenceList title="資料結構" items={evidence.dataStructureEvidence} />
        <EvidenceList title="技巧 / 狀態追蹤" items={evidence.techniqueEvidence ?? []} />
        <EvidenceList title="題型" items={evidence.patternEvidence} />
        <EvidenceList title="常見錯誤" items={evidence.commonMistakes} />
      </div>
    </OutputBlock>
  );
}

function EvidenceList({ title, items }: { title: string; items: string[] }) {
  return (
    <div className="evidence-list">
      <span>{title}</span>
      {items.length > 0 ? <strong>{items.join(", ")}</strong> : <strong>-</strong>}
    </div>
  );
}

function CandidateList({ title, candidates }: { title: string; candidates: TraceCandidate[] }) {
  return (
    <div className="candidate-list">
      <h3>{title}</h3>
      {candidates.length === 0 ? (
        <p className="muted">沒有候選。</p>
      ) : (
        <ol>
          {candidates.slice(0, 4).map((candidate, index) => (
            <li key={candidateKey(title, candidate, index)}>
              <div className="candidate-main">
                <strong>{candidate.title || candidate.id}</strong>
                <span>
                  {[candidate.id, candidate.candidateSource || candidate.source, candidate.problemType]
                    .filter(Boolean)
                    .join(" / ")}
                </span>
                {candidate.concepts && candidate.concepts.length > 0 && (
                  <span>{candidate.concepts.slice(0, 5).join(", ")}</span>
                )}
                {chunkEvidenceBadge(candidate) && (
                  <span className="source-pill">{chunkEvidenceBadge(candidate)}</span>
                )}
                {candidate.rawChunks && candidate.rawChunks.length > 0 && (
                  <details className="payload-details">
                    <summary>原始 chunks ({candidate.rawChunks.length})</summary>
                    <JsonBlock value={candidate.rawChunks} />
                  </details>
                )}
                {candidate.payload && (
                  <details className="payload-details">
                    <summary>Payload</summary>
                    <JsonBlock value={candidate.payload} />
                  </details>
                )}
              </div>
              <b>{formatLabeledScore(candidate.scoreMeta?.displayLabel, candidate.score)}</b>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

function OutputBlock({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="output-block">
      <h2>{title}</h2>
      {children}
    </section>
  );
}

function SimilarProblemCard({ problem }: { problem: SimilarProblem }) {
  return (
    <article className="similar-card">
      <div className="similar-heading">
        <span className="source-pill">{problem.source}</span>
        <h3>
          {problem.id} - {problem.title}
        </h3>
      </div>
      <p>{problem.reason}</p>
      {problem.answerHint && <small>{problem.answerHint}</small>}
      <div className="chips">
        {problem.sharedConcepts.map((concept) => (
          <span key={concept}>{concept}</span>
        ))}
      </div>
    </article>
  );
}

function TwoColumnLists({
  leftTitle,
  leftItems,
  rightTitle,
  rightItems
}: {
  leftTitle: string;
  leftItems: string[];
  rightTitle: string;
  rightItems: string[];
}) {
  return (
    <div className="two-column-lists">
      <OutputBlock title={leftTitle}>
        <OrderedItems items={leftItems} />
      </OutputBlock>
      <OutputBlock title={rightTitle}>
        <OrderedItems items={rightItems} />
      </OutputBlock>
    </div>
  );
}

function OrderedItems({ items }: { items: string[] }) {
  return (
    <ol className="ordered-items">
      {items.map((item) => (
        <li key={item}>{item}</li>
      ))}
    </ol>
  );
}

function ModelPanel({ response }: { response: AnalysisResponse | null }) {
  const config = response?.retrievalConfig;
  const embedding = config?.embeddingProvider;
  const reranker = config?.rerankerProvider;

  return (
    <section className="panel model-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">模型設定</p>
          <h2>Provider / Adapter</h2>
        </div>
      </div>
      <dl className="model-list">
        <div>
          <dt>Embedding</dt>
          <dd>
            {embedding
              ? `${embedding.provider} / ${embedding.adapter || "local"} / ${embedding.model}`
              : (config?.embeddingModel ?? "BAAI/bge-m3")}
          </dd>
        </div>
        <div>
          <dt>Reranker</dt>
          <dd>
            {reranker
              ? `${reranker.provider} / ${reranker.model}`
              : (config?.rerankerModel ?? "BAAI/bge-reranker-v2-m3")}
          </dd>
        </div>
        <div>
          <dt>Language</dt>
          <dd>{config?.language ?? "zh-Hant"}</dd>
        </div>
      </dl>
    </section>
  );
}

function EvidencePathPanel({ paths }: { paths: EvidencePath[] }) {
  return (
    <section className="panel evidence-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">圖證據</p>
          <h2>Graph Evidence</h2>
        </div>
      </div>

      {paths.length === 0 ? (
        <p className="muted">尚無證據路徑。</p>
      ) : (
        paths.map((path) => (
          <article className="path-card" key={path.title}>
            <h3>{path.title}</h3>
            <div className="node-chain">
              {path.nodes.map((node, index) => (
                <span key={`${path.title}-${node.id}-${index}`} className={`node-pill ${node.type}`}>
                  {node.label}
                  {index < path.nodes.length - 1 && <span className="connector">{"->"}</span>}
                </span>
              ))}
            </div>
            <ul className="edge-list">
              {path.edges.map((edge) => (
                <li key={`${path.title}-${edge.from}-${edge.to}-${edge.relation}`}>
                  <span>{edge.relation}</span>
                  <strong>{formatScore(edge.weight)}</strong>
                </li>
              ))}
            </ul>
          </article>
        ))
      )}
    </section>
  );
}

function GraphPathsPanel({ paths }: { paths: GraphPathTrace[] }) {
  return (
    <section className="panel graph-paths-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">圖路徑追蹤</p>
          <h2>Graph Paths</h2>
        </div>
      </div>

      {paths.length === 0 ? (
        <p className="muted">沒有圖路徑。</p>
      ) : (
        paths.map((path, index) => (
          <article className="graph-path-card" key={`graph-path-${index}`}>
            <dl>
              <div>
                <dt>節點</dt>
                <dd>{formatPathPart(path.nodes)}</dd>
              </div>
              <div>
                <dt>關係</dt>
                <dd>{formatPathPart(path.relations)}</dd>
              </div>
              <div>
                <dt>依據</dt>
                <dd>{path.rationale || "-"}</dd>
              </div>
              <div>
                <dt>分數</dt>
                <dd>{formatLabeledScore(path.scoreMeta?.displayLabel, path.score)}</dd>
              </div>
              <div>
                <dt>來源</dt>
                <dd>
                  {graphPathSourceLabel(path.pathSource)} / {graphPathOperationLabel(path.graphPathOperation)}
                </dd>
              </div>
              <div>
                <dt>Path scoring</dt>
                <dd>
                  {path.pathScoring?.strategy ?? "-"} / {formatScore(path.pathScoring?.score)}
                </dd>
              </div>
              <div>
                <dt>Scoring components</dt>
                <dd>{formatPathScoringComponents(path.pathScoring?.components)}</dd>
              </div>
            </dl>
            {path.storePath && (
              <details className="payload-details">
                <summary>Store path</summary>
                <JsonBlock value={path.storePath} />
              </details>
            )}
          </article>
        ))
      )}
    </section>
  );
}

function ContextPanel({ contextPreview }: { contextPreview?: string }) {
  return (
    <section className="panel context-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Context Builder</p>
          <h2>LLM Prompt Context</h2>
        </div>
      </div>
      {contextPreview ? <pre>{contextPreview}</pre> : <p className="muted">Debug mode 關閉。</p>}
    </section>
  );
}
