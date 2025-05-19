from fastapi import APIRouter , Request
from pydantic import BaseModel
from chatbot import generate_response
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class Message(BaseModel):
    prompt: str

async def ask(message: Message, request: Request):
    logger.info(f"Request from {request.client.host}: {message.prompt}")
    response = generate_response(message.prompt)
    logger.info(f"Response length: {len(response)}")
    return {"response": response}