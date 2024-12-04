from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chatbot_logic import process_chat

app = FastAPI()

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    try:
        user_message = chat_request.question
        response = process_chat(user_message)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Chatbot API!"}