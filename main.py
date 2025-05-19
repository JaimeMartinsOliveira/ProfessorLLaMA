from fastapi import FastAPI
import gradio as gr
import uvicorn
from chatbot import generate_response
from api import router as api_router

app = FastAPI()
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

