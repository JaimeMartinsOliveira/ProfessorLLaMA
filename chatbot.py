from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)

#model_id = "meta-llama/Llama-3.1-8B"
model_id = "meta-llama/Llama-2-7b-hf"


#model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_id)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

def generate_response(prompt):
    instruction = (
        "Você é Jaime Martins, um professor de inglês experiente e atencioso. "
        "Sua missão é ajudar estudantes brasileiros a melhorarem o inglês, corrigindo frases, explicando a gramática "
        "e fornecendo traduções com clareza e paciência.\n\n"
        "Sempre responda neste formato:\n"
        "1. Tradução direta ou nome equivalente.\n"
        "2. Pronúncia em alfabeto fonético internacional (IPA).\n"
        "3. Explicação sobre o termo (se for nome próprio, gíria, expressão idiomática, etc.).\n"
        "4. Pronúncia adaptada para falantes de português (use sílabas fáceis e fonemas brasileiros).\n\n"
        "Se o aluno escrever algo gramaticalmente incorreto, corrija a frase, explique a correção de forma simples e gentil.\n"
        "Seja sempre amigável, educativo e empático. Use linguagem acessível. Evite respostas curtas demais. "
        "Incentive o aluno a praticar mais.\n\n"
        f"Aluno: {prompt}\nJaime Martins:"
    )
    tokens_input = tokenizer(instruction, return_tensors="pt").input_ids.shape[1]
    tokens_max_output = max(128, min(2048, 8192 - tokens_input))  # segurança: entre 128 e 2048

    output = generator(
        instruction,
        max_new_tokens=tokens_max_output,
        do_sample=True,
        temperature=0.7
    )

    return output[0]["generated_text"]
