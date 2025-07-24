
# 🤖 Chatbot com LLaMA + FastAPI

Este projeto é uma **API de chatbot** construída com **FastAPI**, utilizando um **modelo de linguagem (LLM)** como o LLaMA ou Mistral, via [Transformers](https://huggingface.co/docs/transformers/) da Hugging Face, com suporte à quantização 4-bit por meio do `bitsandbytes`. Ideal para integração com Web, WhatsApp, aplicativos ou outras interfaces.

---

## 📦 Requisitos

- Python 3.10+ 
- CUDA (opcional, para uso com GPU NVIDIA)
- PyTorch
- transformers 
- bitsandbytes
- accelerate
- uvicorn
- fastapi

### 📥 Instalação

```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuração do Modelo

Este projeto utiliza um modelo LLaMA, Mistral ou similar, com quantização 4-bit para reduzir o uso de memória.

### 🔹 Opção 1: Carregar totalmente na CPU (mais compatível, porém mais lento)

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": "cpu"}
)
```

### 🔹 Opção 2: Offload entre GPU e CPU (exige `accelerate` configurado)

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    llm_int8_enable_fp32_cpu_offload=True
)
```

> 🔗 Consulte a [documentação oficial da Hugging Face](https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu) para mais detalhes sobre o offload.

---

## 🚀 Executando a API

Inicie o servidor local com:

```bash
uvicorn main:app --reload
```

A API estará disponível em: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🔍 Testando a API

Após iniciar, acesse a documentação interativa:

- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### Exemplo de requisição `POST`

```http
POST /chat
Content-Type: application/json

{
  "message": "Olá, tudo bem?"
}
```

---

## 💬 Integrações possíveis

A API pode ser usada com diferentes front-ends e serviços:

- Interfaces Web com **Gradio** ou **Streamlit**
- Bots de WhatsApp usando **Twilio** ou **Venom Bot**
- Aplicativos móveis com **Flutter**, **React Native** ou **Ionic**

---

## 🧠 Modelos Compatíveis

Você pode alterar o modelo via Hugging Face Hub. Exemplos:

```python
model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
```

Outros modelos recomendados:

- `meta-llama/Llama-2-7b-chat-hf`
- `mistralai/Mistral-7B-Instruct-v0.2`
- `tiiuae/falcon-7b-instruct`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (leve e rápido)

---

## 🛠️ Boas práticas e dicas

- Prefira modelos quantizados (`4bit`, `8bit`) para economizar memória.
- Use `device_map={"": "cpu"}` se não tiver GPU ou estiver enfrentando erros de memória.
- Certifique-se de que `accelerate` esteja instalado e atualizado.
- Para notebooks com até **6GB de VRAM**, é altamente recomendado usar quantização com offload para CPU.
