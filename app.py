from fastapi import FastAPI
from pydantic import BaseModel
from utils.workflow import define_workflow
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

class ChatRequest(BaseModel):
    text: str

workflow = define_workflow()
graph = workflow.compile()

@app.post("/chat")
async def chat(request: ChatRequest):
    res = graph.invoke(
        {'messages': [{'role': 'user', 'content': request.text}]}
    )
    return {"response": res['messages'][-1].content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
