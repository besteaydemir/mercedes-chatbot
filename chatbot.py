import utils
import streamlit as st
from streaming import StreamHandler

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

st.set_page_config(page_title="Chatbot", page_icon="💭")
st.header('Chatbot 🤖')
st.write('Hello! I am here to enhance your customer experience. Ask me anything!')

class ContextChatbot:

    def __init__(self):
        self.openai_model = utils.configure_openai()
    
    @st.cache_resource
    def setup_chain(_self):
        memory = ConversationBufferMemory()
        llm = ChatOpenAI(model_name=_self.openai_model, temperature=0, streaming=True)
        chain = ConversationChain(llm=llm, memory=memory, verbose=True)
        return chain
    
    @utils.enable_chat_history
    def main(self):
        chain = self.setup_chain()
        user_query = st.chat_input(placeholder="Ask me anything!")
        
        if st.button('Voice input', key='voice_button', help='This is the voice input button', on_click=lambda: st.write("")):
            user_query = utils.speech_to_text()

        if user_query:
            utils.display_msg(user_query, 'user')
            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                result = chain.invoke(
                    {"input":user_query},
                    {"callbacks": [st_cb]}
                )
                response = result["response"]
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    obj = ContextChatbot()
    obj.main()