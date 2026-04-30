from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{messages}")
])


model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

parser = StrOutputParser()


chain = prompt | model | parser


class State(TypedDict):
  messages: Annotated[list[BaseMessage],add_messages]

def chatbot_node(state:State) -> dict:
  response = chain.invoke({"messages": state["messages"]})

  return {"messages":[response]}

memory = MemorySaver()
config_1 = {"configurable": {"thread_id": "user123"}}
config_2 = {"configurable": {"thread_id": "user456"}}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot",chatbot_node)
graph_builder.add_edge(START,"chatbot")
graph_builder.add_edge("chatbot",END)
graph = graph_builder.compile(checkpointer=memory)

graph.invoke({"messages":"My name is Joe"},config_1)
graph.invoke({"messages":"what's my name"},config_1)
graph.invoke({"messages":"what's my name"},config_2)

# Expect it to know my name in config 1 and not know in 2
# If we dont' do it like a graph then we will be missing out on history