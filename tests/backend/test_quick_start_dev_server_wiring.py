from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_quick_start_passes_custom_backend_origin_to_vite_proxy():
    # Regression: quick-start -BackendPort used to leave Vite proxy on port 8000.
    # Found by /qa on 2026-06-23.
    script = (REPO_ROOT / "scripts" / "quick-start.ps1").read_text(encoding="utf-8")
    vite_config = (REPO_ROOT / "frontend" / "vite.config.ts").read_text(encoding="utf-8")

    assert '$BackendOrigin = "http://127.0.0.1:$BackendPort"' in script
    assert "$env:VITE_BACKEND_ORIGIN = $BackendOriginValue" in script
    assert "Receive-Job -Job $job -ErrorAction SilentlyContinue" in script
    assert "Receive-Job -Job $job -Keep" not in script
    assert "--reload" not in script
    assert "VITE_BACKEND_ORIGIN" in vite_config
    assert '|| "http://127.0.0.1:8000"' in vite_config
