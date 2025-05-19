from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)

model_id = "meta-llama/Llama-3.1-8B"

#model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
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
