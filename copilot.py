from fastapi import FastAPI
from pydantic import BaseModel
from utils.workflow import define_workflow
from copilotkit import LangGraphAGUIAgent 
from ag_ui_langgraph import add_langgraph_fastapi_endpoint
from langgraph.checkpoint.memory import InMemorySaver
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Add CORS middleware before any routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

class ChatRequest(BaseModel):
    text: str

checkpointer = InMemorySaver()
workflow = define_workflow()
graph = workflow.compile(checkpointer=checkpointer)

add_langgraph_fastapi_endpoint(
  app=app,
  agent=LangGraphAGUIAgent(
    name="natdanai_agent", # the name of your agent defined in langgraph.json
    description="AI agent that answers questions about Natdanai Intraraksa, an AI engineer.",
    graph=graph, # the graph object from your langgraph import
  ),
  path="/", # the endpoint you'd like to serve your agent on
)

# @app.post("/chat")
# async def chat(request: ChatRequest):
#     res = graph.invoke(
#         {'messages': [{'role': 'user', 'content': request.text}]},
#         {"configurable": {"thread_id": "1"}}
#     )
#     return {"response": res['messages'][-1].content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("copilot:app",port=8000)
