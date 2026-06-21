import { useRef, useState } from "react";
import type { ReactNode } from "react";
import { fetchAnalysis } from "./api";
import type {
  AnalysisResponse,
  EvidencePath,
  InputKind,
  RequiredConcept,
  RetrievalMode,
  SimilarProblem
} from "./types";

const sampleInput =
  "給定一張無權圖、起點與終點，請找出從起點到終點的最短步數。每次可以沿著一條邊移動，若無法到達請輸出 -1。";

const modes: Array<{ id: RetrievalMode; label: string }> = [
  { id: "hybrid", label: "混合檢索" },
  { id: "vector", label: "向量檢索" },
  { id: "graph", label: "圖譜推理" }
];

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

  return "自動判定";
}

function conceptKindLabel(kind: RequiredConcept["kind"]): string {
  if (kind === "algorithm") {
    return "演算法";
  }

  if (kind === "data_structure") {
    return "資料結構";
  }

  if (kind === "pattern") {
    return "解題模式";
  }

  return "觀念";
}

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export default function App() {
  const [inputText, setInputText] = useState(sampleInput);
  const [mode, setMode] = useState<RetrievalMode>("hybrid");
  const [topK, setTopK] = useState(4);
  const [response, setResponse] = useState<AnalysisResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const latestRequestId = useRef(0);

  async function handleAnalyze() {
    const requestId = latestRequestId.current + 1;
    latestRequestId.current = requestId;
    setIsLoading(true);
    setError(null);

    try {
      const nextResponse = await fetchAnalysis({ inputText, mode, topK });
      if (requestId !== latestRequestId.current) {
        return;
      }

      setResponse(nextResponse);
    } catch (caughtError) {
      if (requestId !== latestRequestId.current) {
        return;
      }

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
          <p className="eyebrow">程式解題知識圖譜 RAG</p>
          <h1>可解釋的演算法與資料結構分析</h1>
        </div>
        <div className="system-status" aria-live="polite">
          <span className={response?.usedMockData ? "status-dot mock" : "status-dot"} />
          {response?.usedMockData ? "本機範例資料" : "API 就緒"}
        </div>
      </header>

      <section className="workspace-grid">
        <aside className="panel input-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">輸入</p>
              <h2>題目或 C++/Python 程式碼</h2>
            </div>
            <button className="ghost-button" type="button" onClick={() => setInputText(sampleInput)}>
              重設範例
            </button>
          </div>

          <textarea
            aria-label="題目或 C++/Python 程式碼"
            value={inputText}
            onChange={(event) => setInputText(event.target.value)}
            placeholder="貼上題目敘述，或貼上包含 BFS / Queue / visited 的 C++、Python 程式碼。"
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
            <label htmlFor="top-k">相似題數量</label>
            <input
              id="top-k"
              min="1"
              max="8"
              type="number"
              value={topK}
              onChange={(event) => setTopK((current) => clampTopK(Number(event.target.value), current))}
            />
          </div>

          <button className="primary-button" type="button" onClick={handleAnalyze} disabled={isLoading}>
            {isLoading ? "分析中..." : "開始分析"}
          </button>

          {error && <p className="error-text">{error}</p>}
        </aside>

        <section className="panel result-panel" aria-live="polite">
          {response ? <AnalysisResult response={response} /> : <EmptyState />}
        </section>

        <section className="side-stack">
          <ModelPanel response={response} />
          <EvidencePanel paths={response?.evidencePaths ?? []} />
        </section>
      </section>
    </main>
  );
}

function EmptyState() {
  return (
    <div className="empty-state">
      <h2>尚未分析</h2>
      <p>貼上題目或程式碼後，系統會輸出題目類型、需要觀念、相似題、相似原因、解題提示與常見錯誤。</p>
    </div>
  );
}

function AnalysisResult({ response }: { response: AnalysisResponse }) {
  return (
    <div className="analysis-result">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">分析結果</p>
          <h2>{inputKindLabel(response.inputKind)}</h2>
        </div>
        <span className="query-id">{response.queryId}</span>
      </div>

      <OutputBlock title="題目類型">
        <p className="problem-type">{response.problemType}</p>
      </OutputBlock>

      <OutputBlock title="需要觀念">
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

      <OutputBlock title="相似題">
        <div className="similar-list">
          {response.similarProblems.map((problem) => (
            <SimilarProblemCard key={`${problem.source}-${problem.id}`} problem={problem} />
          ))}
        </div>
      </OutputBlock>

      <OutputBlock title="為什麼相似">
        <p>{response.similarityReason}</p>
      </OutputBlock>

      <OutputBlock title="解題提示">
        <OrderedItems items={response.solvingHints} />
      </OutputBlock>

      <OutputBlock title="常見錯誤">
        <OrderedItems items={response.commonMistakes} />
      </OutputBlock>
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

  return (
    <section className="panel model-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">檢索設定</p>
          <h2>繁中友善模型</h2>
        </div>
      </div>
      <dl className="model-list">
        <div>
          <dt>嵌入模型</dt>
          <dd>{config?.embeddingModel ?? "BAAI/bge-m3"}</dd>
        </div>
        <div>
          <dt>重排序模型</dt>
          <dd>{config?.rerankerModel ?? "BAAI/bge-reranker-v2-m3"}</dd>
        </div>
        <div>
          <dt>語言</dt>
          <dd>{config?.language ?? "zh-Hant"}</dd>
        </div>
      </dl>
    </section>
  );
}

function EvidencePanel({ paths }: { paths: EvidencePath[] }) {
  return (
    <section className="panel evidence-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">可解釋性</p>
          <h2>證據路徑</h2>
        </div>
      </div>

      {paths.length === 0 ? (
        <p className="muted">分析後會顯示題目、觀念、演算法與資料結構之間的關聯。</p>
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
                  <strong>{formatPercent(edge.weight)}</strong>
                </li>
              ))}
            </ul>
          </article>
        ))
      )}
    </section>
  );
}
