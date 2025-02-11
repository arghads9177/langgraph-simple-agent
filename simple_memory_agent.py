# %% [markdown]
# ### A Simple Arithmatic Tool Calling Agent

# %%
# Import libraries
import os, getpass
from IPython.display import Image, display
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# %% [markdown]
# ### Environment Settings

# %%
# Set environment variables
def _set_env(var:str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")
_set_env("OPENAI_API_KEY")
_set_env("LANGCHAIN_API_KEY")
_set_env("LANGCHAIN_TRACING_V2")
_set_env("LANGCHAIN_ENDPOINT")
_set_env("LANGCHAIN_PROJECT")

# %% [markdown]
# ### Define Tools and LLM and Bind Both

# %%
def calculate(x, y, action= "add") -> float:
    """
    Calculate addition, subtraction, multiplication and division of 2 numbers.

    Agrs:
        a: 1st number
        b: 2nd number
        action: Arithmatic operation
    """
    match action:
        case "add":
            return x + y
        case "subtract":
            return x - y
        case "multiply":
            return  x * y
        case "divide":
            return x / y
        case _:
            return 0

# %%
llm = ChatOpenAI(model="gpt-4o")

# %%
llm_with_tool = llm.bind_tools([calculate])

# %% [markdown]
# ### Define Graph

# %%
# Graph State
class State(MessagesState):
    pass

# %%
sys_msg = SystemMessage(content= "You are a helpful assistant. Perform the arithmatic operations")
# Define node
def tool_calling_llm(state):
    return {"messages": llm_with_tool.invoke([sys_msg] + state["messages"])}

# %%
# Build graph
builder = StateGraph(State)
# Add nodes
builder.add_node("assistant", tool_calling_llm)
builder.add_node("tools", ToolNode([calculate]))
# Add edges
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

# %%
memory = MemorySaver()
graph = builder.compile(checkpointer= memory)

# %%
# View Graph
display(Image(graph.get_graph().draw_mermaid_png()))

# %% [markdown]
# ### Graph Invokation

# %%
# Define thread id
config = { "configurable": { "thread_id": "1"}}

# %%
def invoke_graph(message):
    messages = [HumanMessage(content= message)]
    messages = graph.invoke({"messages": messages}, config)
    for m in messages["messages"]:
        m.pretty_print()

# %%
invoke_graph("Add 8 and 16")

# %%
invoke_graph("Divide that by 6")

# %%
invoke_graph("multiply that by 5")

# %%



