import unittest
import sqlparse
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agent import create_agent
from langgraph.checkpoint.memory import MemorySaver

class TestTooledAgent(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.checkpointer = MemorySaver()
        self.tooled_agent = create_agent(checkpointer=self.checkpointer, llm_model="llama3-70b-8192")

    async def test_agent_workflow_non_database_related(self):
        model_response = await self.tooled_agent.ainvoke(
            {"messages": [HumanMessage(content="Hi. Wer bist du?")]},
            config={
                "configurable": {
                    "thread_id": 1,
                },
            },
        )


        print(model_response["messages"][-2]);
        print(model_response["messages"][-1]);

        # Should be a human message immediately followed by an AIMessage. No tool call needed
        self.assertIsNotNone(model_response["messages"][-1], "The response should not be None")
        self.assertIs(type(model_response["messages"][-1]), AIMessage, "The response content should be an AIMessage")
        self.assertIs(type(model_response["messages"][-2]), HumanMessage, "The query should be a HumanMessage")

    async def test_agent_tool_usage(self):
        model_response = await self.tooled_agent.ainvoke(
            {"messages": [HumanMessage(content="Hi, wie viele Module gibt es?")]},
            config={
                "configurable": {
                    "thread_id": 1,
                },
            },
        )

        print(model_response["messages"][-1].content);

        self.assertIsNotNone(model_response["messages"][-1], "The response should not be None")
        self.assertIs(type(model_response["messages"][-1]), AIMessage, "The response content should be an AIMessage")
        self.assertIs(type(model_response["messages"][-2]), ToolMessage, "Preceding the final AIMessage should be a ToolMessage")
        self.assertIsNotNone(model_response["messages"][-3].tool_calls, "The tool_calls should not be None")

    async def test_agent_generates_valid_SQL(self):
        model_response = await self.tooled_agent.ainvoke(
            {"messages": [HumanMessage(content="What are the names of all the students?")]},
            config={"configurable": {"thread_id": 1}},
        )

        print(model_response["messages"])

        self.assertIs(type(model_response["messages"][-1]), AIMessage, "The response should be an AIMessage")
        self.assertIs(type(model_response["messages"][-2]), ToolMessage, "The response should involve a ToolMessage")

        # Check if the SQL query is syntactically valid
        sql_query = model_response["messages"][-3].tool_calls[0]["args"]["sql_query"]
        is_valid_sql = bool(sqlparse.parse(sql_query))

        self.assertTrue(is_valid_sql, f"The SQL query is not valid: {sql_query}")

    async def test_agent_find_professor(self):
        model_response = await self.tooled_agent.ainvoke(
            {"messages": [HumanMessage(content="Hi, welcher Professor ist für das Modul Datenanalyse zuständig?")]},
            config={
                "configurable": {
                    "thread_id": 1,
                },
            },
        )

        print(model_response["messages"][-1].content);

        self.assertIsNotNone(model_response["messages"][-1], "The response should not be None")
        self.assertIs(type(model_response["messages"][-1]), AIMessage, "The response content should be an AIMessage")
        self.assertIs(type(model_response["messages"][-2]), ToolMessage, "Preceding the final AIMessage should be a ToolMessage")
        self.assertIsNotNone(model_response["messages"][-3].tool_calls, "The tool_calls should not be None")

    async def test_agent_average_marks(self):
        model_response = await self.tooled_agent.ainvoke(
            {"messages": [HumanMessage(content="Hi, welcher Student hat den besten Notendurchschnitt?")]},
            config={
                "configurable": {
                    "thread_id": 1,
                },
            },
        )

        print(model_response["messages"][-1].content);

        self.assertIsNotNone(model_response["messages"][-1], "The response should not be None")
        self.assertIs(type(model_response["messages"][-1]), AIMessage, "The response content should be an AIMessage")
        self.assertIs(type(model_response["messages"][-2]), ToolMessage, "Preceding the final AIMessage should be a ToolMessage")
        self.assertIsNotNone(model_response["messages"][-3].tool_calls, "The tool_calls should not be None")

    async def test_agent_find_best_professor(self):
        model_response = await self.tooled_agent.ainvoke(
            {"messages": [HumanMessage(content="Hi, welcher Professor hat die Prüfung mit dem besten Notendurchschnitt?")]},
            config={
                "configurable": {
                    "thread_id": 1,
                },
            },
        )

        print(model_response["messages"][-1].content);

        self.assertIsNotNone(model_response["messages"][-1], "The response should not be None")
        self.assertIs(type(model_response["messages"][-1]), AIMessage, "The response content should be an AIMessage")
        self.assertIs(type(model_response["messages"][-2]), ToolMessage, "Preceding the final AIMessage should be a ToolMessage")
        self.assertIsNotNone(model_response["messages"][-3].tool_calls, "The tool_calls should not be None")

    async def test_agent_find_unpopular_exams(self):
        model_response = await self.tooled_agent.ainvoke(
            {"messages": [HumanMessage(content="Hi, welche beiden Prüfungen hatten weniger als 3 angemeldete Studenten?")]},
            config={
                "configurable": {
                    "thread_id": 1,
                },
            },
        )

        print(model_response["messages"][-1].content);

        self.assertIsNotNone(model_response["messages"][-1], "The response should not be None")
        self.assertIs(type(model_response["messages"][-1]), AIMessage, "The response content should be an AIMessage")
        self.assertIs(type(model_response["messages"][-2]), ToolMessage, "Preceding the final AIMessage should be a ToolMessage")
        self.assertIsNotNone(model_response["messages"][-3].tool_calls, "The tool_calls should not be None")


if __name__ == '__main__':
    unittest.main()