import gradio as gr
from chatbot import generate_response

def gradio_chat(prompt):
    return generate_response(prompt)

demo = gr.Interface(
    fn=gradio_chat,
    inputs=gr.Textbox(lines=2, placeholder="Ask your English teacher anything..."),
    outputs="text",
    title="English Teacher Chatbot",
    description="Ask questions, get corrections, or improve your grammar!"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
