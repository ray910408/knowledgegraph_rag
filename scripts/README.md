# Scripts

## Quick Start

Run both local services:

```powershell
.\scripts\quick-start.ps1
```

Check prerequisites and paths without starting services:

```powershell
.\scripts\quick-start.ps1 -Check
```

The script starts:

- FastAPI at `http://127.0.0.1:8000`
- Vite at `http://127.0.0.1:5173`

It does not download external datasets or model weights.
