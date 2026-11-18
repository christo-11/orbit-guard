"""Reusable Streamlit presentation components for ORBITGUARD."""

import html

import streamlit as st


def top_navigation(options: tuple[str, ...]) -> str:
    """Render the floating brand and compact horizontal section navigation."""
    with st.container(key="og-nav"):
        brand, links = st.columns([1.8, 4], vertical_alignment="center")
        with brand:
            st.markdown(
                """
                <div class="og-nav-brand">
                  <span class="og-nav-mark">O</span>
                  <span><strong>ORBITGUARD</strong><small>Satellite Telemetry Intelligence</small></span>
                  <span class="og-status" aria-label="Dashboard uses saved project outputs, not live telemetry">
                    <span class="og-status-dot" aria-hidden="true"></span>
                    Saved outputs
                  </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with links:
            selected = st.radio(
                "Main navigation",
                options,
                horizontal=True,
                label_visibility="collapsed",
                key="main_navigation",
            )
    return selected


def section_header(label: str, title: str, copy: str = "") -> None:
    """Render a consistent dashboard section heading."""
    safe_label = html.escape(label)
    safe_title = html.escape(title)
    safe_copy = html.escape(copy)
    copy_html = f'<div class="og-section-copy">{safe_copy}</div>' if copy else ""
    st.markdown(
        f"""
        <div class="og-section">
          <div class="og-section-label">{safe_label}</div>
          <div class="og-section-title">{safe_title}</div>
          {copy_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, kicker: str = "ORBITGUARD / MISSION INTELLIGENCE") -> None:
    """Render the page-level title and product status banner."""
    st.markdown(
        f"""
        <div class="og-hero">
          <div class="og-kicker">{html.escape(kicker)}</div>
          <h1 style="margin:0;">{html.escape(title)}</h1>
          <p>{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def callout(message: str, warning: bool = False) -> None:
    """Render a compact note or expert-review warning."""
    class_name = "og-warning" if warning else "og-callout"
    st.markdown(
        f'<div class="{class_name}">{html.escape(message)}</div>',
        unsafe_allow_html=True,
    )
