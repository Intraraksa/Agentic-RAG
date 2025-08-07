from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, MessagesState,START,END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from typing_extensions import Annotated, Literal,TypedDict, List
from langchain_core.tools import tool
from utils.docs_load import create_vector_store
from utils.workflow import define_workflow

from dotenv import load_dotenv
import os
load_dotenv()

def main(text : str):
    res = graph.invoke(
        {'messages': [{'role': 'user', 'content': text}]}
    )
    print(res['messages'][-1].content)


if __name__ == "__main__":
    workflow = define_workflow()
    graph = workflow.compile()
    text = input("user: ")
    main(text)
