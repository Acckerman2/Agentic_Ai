# %%
from langchain_mistralai import ChatMistralAI
from typing import List,TypedDict,Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


load_dotenv()


# %%
checkpoint = MemorySaver()

# %%
llm = ChatMistralAI()

# %%
#add the messages to the graph

class chatstate (TypedDict): 

    messages: Annotated[List[BaseMessage],add_messages]

# %%
def chat_nodes(state: chatstate):

   messages = state['messages']

   reponse = llm.invoke(messages)

   return {"messages": reponse}




# %%
graph = StateGraph(chatstate)

graph.add_node(chat_nodes,"chat_nodes")


graph.add_edge(START, "chat_nodes")
graph.add_edge("chat_nodes", END)

chat_model = graph.compile(checkpointer=checkpoint)
# %%



