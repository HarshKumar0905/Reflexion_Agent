from dotenv import load_dotenv

load_dotenv()

from langchain_tavily import TavilySearch
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode
from schemas import AnswerQuestion, RevisedAnswer

tavily_tool = TavilySearch(max_results=5)

def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries."""
    return tavily_tool.batch([{"query": query} for query in search_queries])

execute_tool = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=RevisedAnswer.__name__)
    ]
)