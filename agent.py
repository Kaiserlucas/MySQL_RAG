from database_tool import execute_query
from typing import Literal, List, Annotated, TypedDict
from dotenv import load_dotenv
import os
import sqlparse

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

AGENT_SYSTEM_PROMPT = """
You are an advanced AI agent with access to a database. Your task is to answer user questions based on the contents of the database.

You must:
1. Understand the user's query. If the user query does not require a database search, do not perform one and answer the user directly.
2. Be as concise as possible and only answer the user's question
3. Answer in the same language that the user asked the question in

The database schema used is as follows. The first value always denotes the table name, and the other columns are the attributes:

"""

class State(TypedDict):
    messages: Annotated[list, add_messages]

load_dotenv()
api_key = os.getenv("API_KEY")

@tool
def run_database_call(
    sql_query: Annotated[str, "SQL query to run on the database"],
) -> List[dict[str, any]]:
    """
    Executes an SQL query on the database and returns the results as a list of dictionaries.

    Args:
        sql_query (str): The SQL query to execute.

    Returns:
        List[Dict[str, Any]]: The query results, where each row is a dictionary with column names as keys.
    """
    print("SQL Query used is: ")
    print(sql_query)
    result = execute_query(sql_query)

    return result;


def get_last_user_message(state: State) -> HumanMessage:
    messages = state["messages"]
    last_user_message = next(
        (msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None
    )
    return last_user_message


def get_last_tool_message(state: State) -> ToolMessage:
    messages = state["messages"]
    last_tool_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, ToolMessage):
            last_tool_message = msg
            break
    return last_tool_message


def retrieve_schema():
    schema_query = """
        SELECT table_name, GROUP_CONCAT(column_name SEPARATOR ', ') AS columns
        FROM information_schema.columns
        WHERE table_schema NOT IN ('information_schema', 'mysql', 'performance_schema', 'sys')
        GROUP BY table_name;
    """
    return run_database_call.invoke(schema_query)


def is_read_only_sql(query: str) -> bool:
    parsed = sqlparse.parse(query)

    for statement in parsed:
        if statement.get_type() != "SELECT":
            return False

    return True

def create_agent(checkpointer, llm_model="llama3-8b-8192"):
    if api_key is None:
        raise ValueError("API key not found. Set API_KEY as an environment variable.")

    llm_agent = ChatGroq(
        model=llm_model,
        temperature=0,
        max_tokens=250,
        timeout=None,
        max_retries=2,
        api_key=api_key,
    )
    available_tools = [run_database_call]
    llm_agent = llm_agent.bind_tools(available_tools)

    def should_retrieve(state: State) -> Literal["retrieve", END]:
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "retrieve"
        return END

    def agent(state: State):
        schema = "\n".join(f"{table}: {columns}" for table, columns in retrieve_schema())
        total_system_prompt = AGENT_SYSTEM_PROMPT + schema
        query = get_last_user_message(state)

        messages = [SystemMessage(content=total_system_prompt), query]
        response = llm_agent.invoke(messages)
        return {"messages": [response]}

    def generate(state):
        llm_generate = ChatGroq(
            model=llm_model,
            temperature=0,
            timeout=None,
            max_retries=2,
            api_key=api_key,
        )
        messages = state["messages"]
        question = get_last_user_message(state).content
        last_message = messages[-1]
        database_results = last_message.content

        messages = [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(
                content=f"{question}\n\nHere are the database results:\n{database_results}"
            ),
        ]

        response = llm_generate.invoke(messages)
        return {"messages": [response]}

    workflow = StateGraph(State)
    workflow.add_node("agent", agent)
    retrieve = ToolNode(available_tools)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_edge(START, "agent")
    workflow.add_edge("retrieve", "generate")

    workflow.add_conditional_edges("agent", should_retrieve)
    workflow.add_edge("generate", END)

    app = workflow.compile(checkpointer=checkpointer)

    return app


class AppAgent:
    checkpointer = MemorySaver()
    tooled_agent = create_agent(checkpointer=checkpointer)
