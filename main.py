from fastapi import FastAPI
import gradio as gr
import uvicorn
from chatbot import generate_response
from api import router as api_router
import logging
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response


app = FastAPI(
    title="LLaMA Professor API",
    description="API que simula um professor de inglês com LLaMA.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)
app.include_router(api_router)

# Gradio interface
def gradio_chat(prompt):
    return generate_response(prompt)

demo = gr.Interface(
    fn=gradio_chat,
    inputs=gr.Textbox(lines=2, placeholder="Ask your English teacher anything..."),
    outputs="text",
    title="English Teacher Chatbot",
    description="Ask questions, get corrections, or improve your grammar!"
)

# Monta Gradio dentro do FastAPI
@app.get("/")
def gradio_root():
    return {"message": "Go to /gradio to access the chatbot UI."}

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

REQUEST_COUNT = Counter("api_request_count", "Número de requisições à API")

@app.middleware("http")
async def metrics_middleware(request, call_next):
    REQUEST_COUNT.inc()
    response = await call_next(request)
    return response

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
