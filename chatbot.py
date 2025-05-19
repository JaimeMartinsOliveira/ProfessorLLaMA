from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "meta-llama/Llama-3.1-8B"

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

def generate_response(prompt):
    instruction = (
        "You are an English teacher helping a student improve their English. "
        "Correct their sentences, explain grammar, and be friendly.\n\n"
        f"Student: {prompt}\nTeacher:"
    )
    output = generator(instruction, max_length=200, do_sample=True, temperature=0.7)
    return output[0]["generated_text"]
