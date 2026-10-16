"""Bundled web interface, served by the same-origin FastAPI application.

This is a package rather than a folder inside ``sourcebook`` so the frontend and
the retrieval code are independent: each can be read, tested and changed on its
own. ``ROOT`` is the directory holding ``index.html`` and its assets, and it
resolves the same way from a checkout and from an installed wheel.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
