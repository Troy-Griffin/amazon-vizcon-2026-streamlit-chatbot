"""
CityPulse -- Ask CityPulse chatbot (Tier 3: Streamlit + Gemini free API).

Retrieval-grounded: retrieval.py pulls the real commuting numbers for
whichever city/metric a question mentions out of the same data used by
the main CityPulse site's Overview and "A deeper look" tabs, and hands
them to Gemini as context so the model only phrases the answer -- it
should never need to invent a number.

Secrets: set GEMINI_API_KEY in Streamlit Cloud's app secrets (or a local
.streamlit/secrets.toml, gitignored) -- never commit the key to the repo.
"""
import streamlit as st
from google import genai
from google.genai import types

from retrieval import build_context

SYSTEM_PROMPT = (
    "You are the CityPulse commuting assistant, embedded in a data-visualisation site about how "
    "commuting shapes cities (the \"Marchetti constant\" idea: people accept ~1 hour of commuting "
    "a day). Answer questions using ONLY the numeric data provided in the context block for each "
    "turn -- never invent statistics. If the context doesn't contain data for the city asked "
    "about, say so plainly and suggest a city that is covered. Keep answers concise (3-6 "
    "sentences), in the same grounded, narrative tone as the site's \"An Overview\" tab. When you "
    "cite a number, state it plainly (e.g. \"26.8 km/h\")."
)

MODEL_OPTIONS = {
    "Gemini 3.5 Flash-Lite (fastest, highest free quota)": "gemini-3.5-flash-lite",
    "Gemini 3.5 Flash (balanced)": "gemini-3.5-flash",
    "Gemini 3.1 Flash-Lite (alternate fast option)": "gemini-3.1-flash-lite",
}

st.set_page_config(page_title="Ask CityPulse", page_icon="\U0001F4CD", layout="centered")

# ---- Theming to roughly match the main site's dark palette ----
st.markdown(
    """
    <style>
      .stApp { background-color: #05070d; color: #dfe6f5; }
      section[data-testid="stSidebar"] { background-color: #0d1220; }
      .stChatMessage { background-color: #0d1220; border: 1px solid #1c2438; border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Ask CityPulse")
st.caption(
    "Ask about commute speed, time, distance, affordability or accessibility for any city in the "
    "dataset. Answers are grounded in the same real commuting data behind the Overview and "
    "\"A deeper look\" tabs, phrased by Google's Gemini API (free tier)."
)

with st.sidebar:
    st.subheader("Settings")
    model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
    model_id = MODEL_OPTIONS[model_label]

    api_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
    if not api_key:
        api_key = st.text_input("Gemini API key", type="password", help="Get a free key at aistudio.google.com/apikey")

    st.markdown("---")
    
    if st.button("Clear chat"):
        st.session_state.pop("messages", None)
        st.rerun()

if not api_key:
    st.info("Enter a Gemini API key in the sidebar (or configure GEMINI_API_KEY in secrets) to start chatting.")
    st.stop()

try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error(f"Failed to initialize Gemini client: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("cities_used"):
            st.caption("Data used: " + ", ".join(msg["cities_used"]))

user_input = st.chat_input("e.g. How does Tokyo's commute compare to London's?")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    retrieval = build_context(user_input)
    prompt = f"Context:\n{retrieval['contextText']}\n\nQuestion: {user_input}"

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""
        try:
            stream = client.models.generate_content_stream(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.4,
                ),
            )
            for chunk in stream:
                if chunk.text:
                    full_text += chunk.text
                    placeholder.markdown(full_text)
        except Exception as e:
            full_text = f"Sorry, something went wrong calling Gemini: {e}"
            placeholder.markdown(full_text)

        if retrieval["matchedCities"]:
            st.caption("Data used: " + ", ".join(retrieval["matchedCities"]))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_text,
            "cities_used": retrieval["matchedCities"],
        }
    )
