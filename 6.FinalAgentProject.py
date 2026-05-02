from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langgraph.graph import StateGraph,END
from typing import TypedDict,Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage

memory = MemorySaver()
config = {"configurable": {"thread_id": "user123"}}

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a helpful assistant.
    You have access to:
    - rag_search → for questions about the provided document/book
    - DuckDuckGo → for general or real-world knowledge

    Choose the appropriate tool when needed.
    """),
    MessagesPlaceholder("messages")
])

# Model
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# RAG 
loader = TextLoader("84.txt.utf-8")
docs = loader.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_documents(docs)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "Answer ONLY from the provided context.\n"
     "If the answer is not in the context, say 'I don't know'.\n\n{context}"
    ),
    ("human", "{input}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": (lambda x: x["input"]) | retriever | format_docs,
        "input": lambda x: x["input"]
    }
    | rag_prompt
    | model
    | StrOutputParser()
)

@tool 
def rag_search(query: str) -> str:
    """
    Search and answer questions about the loaded book/document.
    Use this ONLY when the question is about the document content.
    Do NOT use for general knowledge or current events.
    """
    return rag_chain.invoke({"input": query})

# Tools Search + RAG
search_tool = DuckDuckGoSearchRun()
tools = [search_tool,rag_search]

model_with_tools = model.bind_tools(tools)

# Chain 
chain = prompt | model_with_tools

# Node
def llm_with_tools(state):
    response = chain.invoke({"messages": state["messages"]})
    return {"messages": [response]}

# Agent State
class AgentState(TypedDict):
  messages: Annotated[list[BaseMessage],add_messages]

# Graph
graph = StateGraph(AgentState)

graph.add_node("llm", llm_with_tools)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("llm")
graph.add_conditional_edges("llm", tools_condition)
graph.add_edge("tools", "llm")

app = graph.compile(checkpointer=memory)

def format_output(state):
    ai_messages = [m for m in state["messages"] if isinstance(m, AIMessage)]
    return {"final_answer": ai_messages[-1].content}

response = app.invoke({"messages":"Who is the creature from Frankstien novel context and from where did you answer this"},config)
format_output(response)

