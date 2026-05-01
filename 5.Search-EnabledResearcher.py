from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langgraph.graph import StateGraph,END
from typing import TypedDict,Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolNode,tools_condition

memory = MemorySaver()
config = {"configurable": {"thread_id": "user123"}}

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("messages")
])

# Model
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# Tools
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

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

reponse = app.invoke({"messages":"Search the internet and tell me what's the result of last PSG vs Bayern game"},config)
final_result = reponse[-1]['content']
print(final_result)
