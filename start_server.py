"""Convenience launcher for the ClaimCheck AI API.

Equivalent to `uvicorn api.server:app --reload`, but runnable as a plain script:

    python start_server.py

The API serves on http://localhost:8000 (docs at /docs). Run the frontend
separately with `cd frontend && npm run dev` (http://localhost:3000).
"""

from __future__ import annotations

import uvicorn

if __name__ == "__main__":
    uvicorn.run("api.server:app", host="127.0.0.1", port=8000, reload=True)
