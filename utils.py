import os
import openai
import speech_recognition as sr
import streamlit as st
from datetime import datetime

#decorator
def enable_chat_history(func): 
    if "messages" not in st.session_state:
        st.session_state.messages = []
    current_page = func.__qualname__
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = current_page
    if st.session_state["current_page"] != current_page:
        try:
            st.cache_resource.clear()
            del st.session_state["current_page"]
            del st.session_state["messages"]
        except:
            pass

        # to show chat history on ui
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

    def execute(*args, **kwargs):
        func(*args, **kwargs)
    return execute

def display_msg(msg, author):
    """Method to display message on the UI

    Args:
        msg (str): message to display
        author (str): author of the message -user/assistant
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)

def configure_openai():
    openai_api_key = "sk-proj-o2VOoBtB0B6zIvKuYi1kT3BlbkFJt91XJ6lTjVxoHNhJnXVa"
    print(openai_api_key)
    if openai_api_key:
        st.session_state['OPENAI_API_KEY'] = openai_api_key
        os.environ['OPENAI_API_KEY'] = openai_api_key
    else:
        st.error("Please add your OpenAI API key to continue.")
        st.info("Obtain your key from this link: https://platform.openai.com/account/api-keys")
        st.stop()

    model = "gpt-3.5-turbo"
    try:
        client = openai.OpenAI()
        available_models = [{"id": i.id, "created":datetime.fromtimestamp(i.created)} for i in client.models.list() if str(i.id).startswith("gpt")]
        available_models = sorted(available_models, key=lambda x: x["created"])
        available_models = [i["id"] for i in available_models]

        model = st.sidebar.selectbox(
            label="Model",
            options=available_models,
            index=available_models.index(st.session_state['OPENAI_MODEL']) if 'OPENAI_MODEL' in st.session_state else 0
        )
        st.session_state['OPENAI_MODEL'] = model
    except openai.AuthenticationError as e:
        st.error(e.body["message"])
        st.stop()
    except Exception as e:
        print(e)
        st.error("Something went wrong. Please try again later.")
        st.stop()
    return model


def speech_to_text():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    with microphone as source:
        print("Say something!")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio, language="en-US")
        return text.lower()  
    except sr.UnknownValueError:
        st.write("Sorry, I could not understand what you said.")
        return None
    except sr.RequestError as e:
        st.write(f"Could not request results from Google Speech Recognition service; {e}")
        return None
    