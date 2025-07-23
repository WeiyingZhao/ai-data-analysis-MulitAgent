import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain.agents import load_tools
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import HumanMessage

from create_agent import create_agent, create_supervisor, create_note_agent
from node import (
    agent_node,
    human_choice_node,
    note_agent_node,
    human_review_node,
    refiner_node,
)
from router import QualityReview_router, hypothesis_router, process_router
from state import State
from prompts import load_prompt
from tools.internet import google_search, FireCrawl_scrape_webpages
from tools.basetool import execute_code, execute_command
from tools.FileEdit import create_document, read_document, edit_document, collect_data
from logger import setup_logger

load_dotenv()
logger = setup_logger()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
WORKING_DIRECTORY = os.getenv("WORKING_DIRECTORY", "./data_storage/")


def build_workflow(llm: ChatOpenAI, power_llm: ChatOpenAI, json_llm: ChatOpenAI) -> StateGraph:
    """Build the LangGraph workflow."""
    members = [
        "Hypothesis",
        "Process",
        "Visualization",
        "Search",
        "Coder",
        "Report",
        "QualityReview",
        "Refiner",
    ]

    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    hypothesis_agent = create_agent(
        llm,
        [collect_data, wikipedia, google_search, FireCrawl_scrape_webpages]
        + load_tools(["arxiv"]),
        load_prompt("hypothesis.txt"),
        members,
        WORKING_DIRECTORY,
    )

    process_agent = create_supervisor(
        power_llm,
        load_prompt("supervisor.txt"),
        ["Visualization", "Search", "Coder", "Report"],
    )

    visualization_agent = create_agent(
        llm,
        [read_document, execute_code, execute_command],
        load_prompt("visualization.txt"),
        members,
        WORKING_DIRECTORY,
    )

    code_agent = create_agent(
        power_llm,
        [read_document, execute_code, execute_command],
        load_prompt("code.txt"),
        members,
        WORKING_DIRECTORY,
    )

    searcher_agent = create_agent(
        llm,
        [read_document, collect_data, wikipedia, google_search, FireCrawl_scrape_webpages]
        + load_tools(["arxiv"]),
        load_prompt("search.txt"),
        members,
        WORKING_DIRECTORY,
    )

    report_agent = create_agent(
        power_llm,
        [create_document, read_document, edit_document],
        load_prompt("report.txt"),
        members,
        WORKING_DIRECTORY,
    )

    quality_review_agent = create_agent(
        llm,
        [create_document, read_document, edit_document],
        load_prompt("quality_review.txt"),
        members,
        WORKING_DIRECTORY,
    )

    note_agent = create_note_agent(
        json_llm,
        [read_document],
        load_prompt("note.txt"),
    )

    refiner_agent = create_agent(
        power_llm,
        [read_document, edit_document, create_document, collect_data, wikipedia, google_search, FireCrawl_scrape_webpages]
        + load_tools(["arxiv"]),
        load_prompt("refiner.txt"),
        members,
        WORKING_DIRECTORY,
    )

    workflow = StateGraph(State)

    workflow.add_node("Hypothesis", lambda s: agent_node(s, hypothesis_agent, "hypothesis_agent"))
    workflow.add_node("Process", lambda s: agent_node(s, process_agent, "process_agent"))
    workflow.add_node("Visualization", lambda s: agent_node(s, visualization_agent, "visualization_agent"))
    workflow.add_node("Search", lambda s: agent_node(s, searcher_agent, "searcher_agent"))
    workflow.add_node("Coder", lambda s: agent_node(s, code_agent, "code_agent"))
    workflow.add_node("Report", lambda s: agent_node(s, report_agent, "report_agent"))
    workflow.add_node("QualityReview", lambda s: agent_node(s, quality_review_agent, "quality_review_agent"))
    workflow.add_node("NoteTaker", lambda s: note_agent_node(s, note_agent, "note_agent"))
    workflow.add_node("HumanChoice", human_choice_node)
    workflow.add_node("HumanReview", human_review_node)
    workflow.add_node("Refiner", lambda s: refiner_node(s, refiner_agent, "refiner_agent"))

    workflow.add_edge("Hypothesis", "HumanChoice")
    workflow.add_conditional_edges(
        "HumanChoice",
        hypothesis_router,
        {"Hypothesis": "Hypothesis", "Process": "Process"},
    )

    workflow.add_conditional_edges(
        "Process",
        process_router,
        {
            "Coder": "Coder",
            "Search": "Search",
            "Visualization": "Visualization",
            "Report": "Report",
            "Process": "Process",
            "Refiner": "Refiner",
        },
    )

    for member in ["Visualization", "Search", "Coder", "Report"]:
        workflow.add_edge(member, "QualityReview")

    workflow.add_conditional_edges(
        "QualityReview",
        QualityReview_router,
        {
            "Visualization": "Visualization",
            "Search": "Search",
            "Coder": "Coder",
            "Report": "Report",
            "NoteTaker": "NoteTaker",
        },
    )
    workflow.add_edge("NoteTaker", "Process")
    workflow.add_edge("Refiner", "HumanReview")
    workflow.add_conditional_edges(
        "HumanReview",
        lambda state: "Process" if state and state.get("needs_revision", False) else "END",
        {"Process": "Process", "END": END},
    )

    workflow.add_edge(START, "Hypothesis")

    return workflow


def run_workflow(user_input: str):
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    power_llm = ChatOpenAI(model="gpt-4o")
    json_llm = ChatOpenAI(model="gpt-4o")

    workflow = build_workflow(llm, power_llm, json_llm)
    graph = workflow.compile()

    events = graph.stream(
        {
            "messages": [HumanMessage(content=user_input)],
            "hypothesis": "",
            "process_decision": "",
            "process": "",
            "visualization_state": "",
            "searcher_state": "",
            "code_state": "",
            "report_section": "",
            "quality_review": "",
            "needs_revision": False,
            "last_sender": "",
        },
        {"configurable": {"thread_id": "1"}, "recursion_limit": 3000},
        stream_mode="values",
        debug=False,
    )
    for s in events:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message, end="", flush=True)
        else:
            print(message)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the multi-agent workflow")
    parser.add_argument("prompt", help="User input prompt for the workflow")
    args = parser.parse_args()
    run_workflow(args.prompt)
