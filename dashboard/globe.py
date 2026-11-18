"""Streamlit wrapper for the standalone illustrative 3D globe."""

from pathlib import Path

import streamlit.components.v1 as components


GLOBE_HTML = Path(__file__).with_name("globe.html")


def render_globe(height: int = 480) -> None:
    """Render the self-contained animated canvas globe."""
    if not GLOBE_HTML.is_file():
        raise FileNotFoundError(f"3D globe asset is missing: {GLOBE_HTML}")
    components.html(
        GLOBE_HTML.read_text(encoding="utf-8"),
        height=height,
        scrolling=False,
    )
