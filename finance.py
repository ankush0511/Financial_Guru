from langchain_core.messages import ChatMessage
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools.google_finance import GoogleFinanceQueryRun
from langchain_community.utilities.google_finance import GoogleFinanceAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType,load_tools
from langchain.callbacks import StreamlitCallbackHandler
import os
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain.schema import Document
from typing import Iterable
from dotenv import load_dotenv
from streamlit.runtime.state import session_state
load_dotenv()
LANGCHAIN_API_KEY=st.secrets["LANGCHAIN_API_KEY"]
GEMMINI_API_LEY=st.secrets["GEMMINI_API_LEY"]
GROQ_API_KEY=st.secrets["GROQ_API_KEY"]
model = ChatGroq(model='Gemma2-9b-It')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ["LANGCHAIN_PROJECT"] = "Finance"
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['SERP_API_KEY']=os.getenv('SERP_API_KEY')
class CustomYahooFinanceNewsTool(YahooFinanceNewsTool):
  @staticmethod
  def _format_results(docs: Iterable[Document], query: str) -> str:
      doc_strings = []
      for doc in docs:
          if "description" in doc.metadata and "title" in doc.metadata:
              if query in doc.metadata["description"] or query in doc.metadata["title"]:
                  doc_strings.append(
                      "\n".join([doc.metadata["title"], doc.metadata["description"]])
                  )
          else:
              description = doc.metadata.get("description", "")
              title = doc.metadata.get("title", "")  # Get title, default to empty string
              if query in description or query in title:
                  doc_strings.append("\n".join([title, description]))
      return "\n\n".join(doc_strings)
search = DuckDuckGoSearchRun(name='search')
llm = ChatGroq(model_name="Llama3-8b-8192", streaming=True)
#google finance
st.title('Chat with Financial Guru')
SERP_API_KEY=st.text_input("Enter a SERP_API_KEY", type="password")
if SERP_API_KEY:
    googlef=GoogleFinanceAPIWrapper(serp_api_key=SERP_API_KEY)
    googlefinance=GoogleFinanceQueryRun(api_wrapper=googlef)
    yahhof=CustomYahooFinanceNewsTool(top_k_results=1, doc_content_chars_max=500)
    # latest_query = prompt
    if "messages" not in st.session_state:
      st.session_state["messages"] = [{
          "role":
          "user",
          "content":
          "Hii i am a Chatbot Guru who can search the Internet. How can i help you to become Rich?"
      }]
    for msg in st.session_state.messages:
      st.chat_message(msg["role"]).write(msg["content"])
    
    
    if prompt := st.chat_input(placeholder="what is machine learning"):
      st.session_state.messages.append({"role": "user", "content": prompt})
      st.chat_message("user").write(prompt)    
      tools = [yahhof,googlefinance,search]
    
      search_agent = initialize_agent(tools,
                                      llm,
                                      agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                      handling_parsing_error=True,
                                      verbose=True)
      with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        response = search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
        st.write(response)
else:
    st.error("Please enter a SERP_API_KEY.")


