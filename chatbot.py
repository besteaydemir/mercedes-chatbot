import utils
import streamlit as st
from streaming import StreamHandler

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from utils import SalesGPT
import os

st.set_page_config(page_title="Chatbot", page_icon="💭")
st.header('Chatbot 🤖')
st.write('Hello! I am here to enhance your customer experience. Ask me anything!')

class ContextChatbot:

    def __init__(self):
        # self.openai_model = utils.configure_openai()
        # chain = self.setup_chain()
        os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

        llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        
        self.conversation_stages = {
            "1": "Introduction: Start the conversation by introducing yourself and your company. Be polite and respectful while keeping the tone of the conversation professional. Your greeting should be welcoming. Always clarify in your greeting the reason why you are contacting the prospect.",
            "2": "Qualification: Qualify the prospect by confirming if they are the right person to talk to regarding your product/service. Ensure that they have the authority to make purchasing decisions.",
            "3": "Value proposition: Briefly explain how your product/service can benefit the prospect. Focus on the unique selling points and value proposition of your product/service that sets it apart from competitors.",
            "4": "Needs analysis: Ask open-ended questions to uncover the prospect's needs and pain points. Listen carefully to their responses and take notes.",
            "5": "Solution presentation: Based on the prospect's needs, present your product/service as the solution that can address their pain points.",
            "6": "Objection handling: Address any objections that the prospect may have regarding your product/service. Be prepared to provide evidence or testimonials to support your claims.",
            "7": "Close: Ask for the sale by proposing a next step. This could be a demo, a trial or a meeting with decision-makers. Ensure to summarize what has been discussed and reiterate the benefits.",
        }
        self.config = dict(
            salesperson_name = "John",
            salesperson_role = "Salesman",
            company_name = "Mercedes",
            company_business = "Mercedes is a famous car company which sells premium cars having the latest technology. The price range ranges from mid to high priced",
            company_values = """You should always recommend a Mercedes Benz electric vehicle.
            -Provide benefits of electric vehicles. If the user likes a non-electric vehicle, recommend the most similar electric vehicle.
            -Address specific concerns about the range, price, etc. based on the information tool.
            -Ask users about their preferences, such as budget, driving habits and desired features. Highlight the key features.
            -Provide comparisons between different ev models and traditional gas powered vehicles. Make positive remarks about the electric cars.
            -Emphasize long term benefits of an EV.
            - Ask the users what is the most important for them in their consideration. Allow them to pick from emission, range problems, dimension, price.  Talk more about what is important for the customer user.""",
            conversation_purpose = "find out which electric vehicle is more suited for the customer's use case and recommend them as such. at no case should you recommend petrol or hybrid cars.",
            conversation_history=[],
            conversation_type="call",
            conversation_stage=self.conversation_stages.get(
                "1",
                "Introduction: Start the conversation by introducing yourself and your company. Be polite and respectful while keeping the tone of the conversation professional.",
            ),
            use_tools=True,
            product_catalog="Products_info.txt",
        )

        self.openai_model = SalesGPT.from_llm(llm, verbose=False, **self.config)

    @st.cache_resource
    def setup_chain(_self):
        memory = ConversationBufferMemory()
        llm = ChatOpenAI(model_name=_self.openai_model, temperature=0, streaming=True)
        chain = ConversationChain(llm=llm, memory=memory, verbose=True)
        return chain
    
    @utils.enable_chat_history
    def main(self):

        
        user_query = st.chat_input(placeholder="Ask me anything!")
        
        if st.button('Voice input', key='voice_button', help='This is the voice input button', on_click=lambda: st.write("")):
            user_query = utils.speech_to_text()

        if user_query:
            utils.display_msg(user_query, 'user')
            self.openai_model.human_step(
                    user_query
                )
            with st.chat_message("assistant"):
                #st_cb = StreamHandler(st.empty())
                st_cb = st.empty()
                cid =  self.openai_model.determine_conversation_stage()
                response = self.openai_model.step()
                response = response.split(":")
                #print(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st_cb.write(f'{response[-1]}')

if __name__ == "__main__":
    obj = ContextChatbot()
    obj.main()