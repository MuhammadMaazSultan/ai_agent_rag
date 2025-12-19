import streamlit as st
from config import MODEL_NAME, EMBEDDING_MODEL, PROMPT_TEMPLATE
from chatbot import HistoryChatBotRag

st.set_page_config(
    page_title="History Chatbot",
    page_icon = 'History',
    layout = 'centered',
)
st.title("📜 History Chatbot")
st.markdown("""
This is a simple chatbot that can answer your history-related questions.""")

## Initialize the HistoryBot 
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = HistoryChatBotRag(MODEL_NAME,EMBEDDING_MODEL, PROMPT_TEMPLATE)

## Memory Initialization
if 'message' not in st.session_state:
    st.session_state.message = []

for message in st.session_state.message:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if user_input := st.chat_input('Ask me anything about history or type exit for new chat'):
    st.session_state.message.append({'role':'user', 'content':user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    if user_input.strip().lower() == 'exit':
        st.session_state.message = []
        st.session_state.chatbot.clear_memory()
        st.rerun()
    else:
        with st.chat_message('historybot'):
            with st.spinner('HistoryBot is thinking...'):
                response = st.session_state.chatbot.get_response(user_input)
            st.markdown(response)
        st.session_state.message.append({'role':'historybot', 'content':response})
        