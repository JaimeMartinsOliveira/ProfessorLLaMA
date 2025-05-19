from fastapi import APIRouter
from pydantic import BaseModel
from chatbot import generate_response

router = APIRouter()

class Message(BaseModel):
    prompt: str

@router.post("/ask")
def ask(message: Message):
    return {"response": generate_response(message.prompt)}
