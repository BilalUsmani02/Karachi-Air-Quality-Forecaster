"""Pure, Streamlit-free helpers behind `app/` (Phase 7, spec §9-10).

Every module here can be imported and tested without a Streamlit runtime: no `import
streamlit`, no widgets, no caching decorators. The `app/` layer calls these functions and
wraps the slow ones (`src.ui.explain_ui`) in `st.cache_resource` / `st.cache_data` itself.
"""
