import streamlit as st

# ========================
# Page Setup
# ========================
st.set_page_config(
    page_title="Mental Health Chatbot",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Mental Health Support Chatbot")


# ========================
# Load backend safely (IMPORTANT FIX)
# ========================
@st.cache_resource
def load_models():

    from predict import predict_emotion
    from gpt_handler import get_gpt_response

    return predict_emotion, get_gpt_response


predict_emotion, get_gpt_response = load_models()


# ========================
# Conversation Memory
# ========================
if "messages" not in st.session_state:

    st.session_state.messages = []


# Display previous messages
for role, message in st.session_state.messages:

    st.chat_message(role).write(message)


# ========================
# User Input
# ========================
user_text = st.chat_input("Type your message")


if user_text:

    # Show user message
    st.chat_message("user").write(user_text)


    # Emotion detection
    emotion = predict_emotion(user_text)


    # Prompt creation
    prompt = f"""
You are a supportive mental health assistant.

Detected emotion: {emotion}

User message:
{user_text}

Respond with empathy and support.
"""


    # GPT response
    reply = get_gpt_response(prompt)


    # Save messages
    st.session_state.messages.append(("user", user_text))
    st.session_state.messages.append(("assistant", reply))


    # Show detected emotion
    # st.info(f"Detected emotion: {emotion}")


    # Show bot reply
    st.chat_message("assistant").write(reply)