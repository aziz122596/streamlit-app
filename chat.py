import streamlit as st
from streamlit_chat import message

st.title("Тест компонента streamlit_chat")
message("Привет от бота", is_user=False)
message("Привет, бот!", is_user=True)
