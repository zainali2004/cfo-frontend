"""
FrontEnd/main.py
Entry point — immediately redirects to the Chatbot POC.

The chatbot page (pages/chatbot.py) is the sole UI for this POC.
This file exists only so users can start the app with the familiar command:

    streamlit run main.py

st.switch_page() hands control to pages/chatbot.py before any content
renders here, so the user never sees this file.
"""

import streamlit as st

st.set_page_config(
    page_title="Data 2 Insight",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.switch_page("pages/chatbot.py")
