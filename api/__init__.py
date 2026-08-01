"""HTTP interface for the ClaimCheck AI pipeline.

`api/server.py` exposes the same pipeline that `main.py` drives from the CLI, as
a FastAPI application. It imports and calls the existing pipeline functions — it
does not reimplement any pipeline logic.
"""
