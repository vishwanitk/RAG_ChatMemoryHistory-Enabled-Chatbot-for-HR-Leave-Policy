import streamlit as st
from chatbot4 import with_message_history, get_session_history

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="HR Leave Policy Assistant",
    page_icon="📄",
    layout="centered"
)

st.title("📄 HR Leave Policy Assistant")
st.caption("Ask questions strictly based on the leave policy document")

# -------------------------------
# Session handling
# -------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit_session_001"

config = {
    "configurable": {
        "session_id": st.session_state.session_id
    }
}

# -------------------------------
# Display chat history
# -------------------------------
history = get_session_history(st.session_state.session_id)

for msg in history.messages:
    role = "assistant" if msg.type == "ai" else "user"
    with st.chat_message(role):
        st.markdown(msg.content)

# -------------------------------
# Chat input
# -------------------------------

user_input = st.chat_input("Your Question on sick/casual/maternity leaves...")

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get model response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = with_message_history.invoke(
                {"question": user_input},
                config=config
            )
            st.markdown(response.content)
