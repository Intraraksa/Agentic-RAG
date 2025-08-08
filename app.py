from fastapi import FastAPI
from pydantic import BaseModel
from utils.workflow import define_workflow
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
    expose_headers=["*"],)


class ChatRequest(BaseModel):
    text: str

checkpointer = InMemorySaver()
workflow = define_workflow()
graph = workflow.compile()

@app.post("/chat")
async def chat(request: ChatRequest):
    res = graph.invoke(
        {'messages': [{'role': 'user', 'content': request.text}]},
        # {"configurable": {"thread_id": "1"}}
    )
    return {"response": res['messages'][-1].content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
