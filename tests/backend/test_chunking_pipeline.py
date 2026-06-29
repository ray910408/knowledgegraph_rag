from __future__ import annotations

from pathlib import Path

from backend.app.chunking.code import CodeChunker
from backend.app.chunking.fallback import GenericFallbackChunker
from backend.app.chunking.problem_statement import ProblemStatementChunker


def test_problem_statement_chunker_splits_english_headings():
    text = """
Description
Find the shortest path.

Input
The first line contains n.

Output
Print the answer.

Sample Input
3

Sample Output
7
""".strip()

    chunks = ProblemStatementChunker().chunk(text, input_id="raw-en")
    kinds = [chunk.kind for chunk in chunks]

    assert kinds == ["statement", "input", "output", "examples"]
    examples_chunk = chunks[-1]
    assert "Sample Input" in examples_chunk.display_text
    assert "Sample Output" in examples_chunk.display_text
    assert all(chunk.display_text.strip() for chunk in chunks)


def test_problem_statement_chunker_splits_chinese_headings():
    text = """
題目敘述
找出最短路徑。

輸入
第一行給 n。

輸出
輸出答案。

範例
輸入 3
輸出 7
""".strip()

    chunks = ProblemStatementChunker().chunk(text, input_id="raw-zh")
    kinds = [chunk.kind for chunk in chunks]

    assert kinds == ["statement", "input", "output", "examples"]
    assert "輸入 3" in chunks[-1].display_text
    assert "輸出 7" in chunks[-1].display_text
    assert all(chunk.display_text.strip() for chunk in chunks)


def test_problem_statement_chunker_falls_back_to_paragraph_chunks():
    text = "first paragraph\n\nsecond paragraph\n\nthird paragraph"

    chunks = ProblemStatementChunker().chunk(text, input_id="raw-paragraph")

    assert [chunk.kind for chunk in chunks] == ["statement", "statement", "statement"]
    assert [chunk.display_text for chunk in chunks] == [
        "first paragraph",
        "second paragraph",
        "third paragraph",
    ]


def test_generic_fallback_chunker_returns_at_least_one_chunk():
    chunks = GenericFallbackChunker().chunk("noise noise noise", input_id="unknown-1")

    assert len(chunks) >= 1
    assert all(chunk.display_text.strip() for chunk in chunks)


def test_code_chunker_extracts_cpp_main_or_function():
    text = """
    #include <bits/stdc++.h>
    using namespace std;

    int bfs() {
        return 1;
    }

    int main() {
        return bfs();
    }
    """.strip()

    chunks = CodeChunker().chunk(text, input_id="code-cpp")

    assert any("main" in str(chunk.metadata.get("symbolName", "")) for chunk in chunks) or any(
        "int main" in chunk.display_text for chunk in chunks
    )


def test_code_chunker_detects_supported_languages():
    chunker = CodeChunker()

    assert chunker.detect_language("#include <iostream>\nint main(){}") == "cpp"
    assert chunker.detect_language("def solve():\n    return 1") == "python"
    assert chunker.detect_language("function solve() { return 1 }") == "javascript"
    assert chunker.detect_language(
        "public class Main { public static void main(String[] args) {} }"
    ) == "java"


def test_code_chunker_falls_back_to_line_window_chunks():
    text = "x = 1\n\ny = 2\n\nz = 3\n"

    chunks = CodeChunker().chunk(text, input_id="code-fallback")

    assert len(chunks) >= 1
    assert any(chunk.kind == "code_window" for chunk in chunks)


def test_code_chunker_never_emits_empty_chunk():
    chunks = CodeChunker().chunk("def solve():\n    pass\n", input_id="code-non-empty")

    assert chunks
    assert all(chunk.display_text.strip() for chunk in chunks)


def test_lab_chunkers_are_not_wired_into_runtime():
    repo_root = Path(__file__).resolve().parents[2]
    ingestion_text = (repo_root / "backend" / "app" / "ingestion" / "pipeline.py").read_text(
        encoding="utf-8"
    )
    retrieval_text = (repo_root / "backend" / "app" / "retrieval" / "pipeline.py").read_text(
        encoding="utf-8"
    )

    for marker in ("CodeChunker", "ProblemStatementChunker", "GenericFallbackChunker"):
        assert marker not in ingestion_text
        assert marker not in retrieval_text
