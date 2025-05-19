
# 🤖 ChatBot com LLaMA e FastAPI

Este projeto é uma API de chatbot construída com **FastAPI**, utilizando um modelo de linguagem grande (LLM) como o **LLaMA** via `transformers` da Hugging Face e suporte a quantização com `bitsandbytes`. Ideal para uso em aplicações com interface via Web, WhatsApp ou integração com outros serviços.

---

## 📦 Requisitos

- Python 3.10 ou superior
- CUDA (opcional, para uso com GPU NVIDIA)
- PyTorch
- `transformers`
- `bitsandbytes`
- `accelerate`
- `uvicorn`
- `fastapi`

Instale com:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuração do modelo

Este projeto usa um modelo LLaMA ou similar com quantização 4-bit para reduzir o uso de memória.

**Atenção:** Se sua GPU não tiver memória suficiente, o carregamento do modelo poderá falhar. Duas abordagens estão disponíveis:

### ✅ Opção 1: Carregar o modelo inteiramente na CPU (mais compatível, mais lento)

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": "cpu"}
)
```

### ✅ Opção 2: Permitir offload entre GPU e CPU (requer `accelerate` configurado)

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    llm_int8_enable_fp32_cpu_offload=True
)
```

> Consulte [a documentação oficial](https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu) da Hugging Face para mais detalhes sobre offload.

---

## 🚀 Executando a API

Para rodar localmente com **FastAPI + Uvicorn**:

```bash
uvicorn main:app --reload
```

A aplicação será iniciada em: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🔍 Testando a API

Após iniciar a aplicação, acesse a documentação interativa:

- Swagger: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Redoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

Você pode enviar mensagens ao chatbot via `POST`:

```json
POST /chat
{
  "message": "Olá, tudo bem?"
}
```

---

## 💻 Exemplo de uso com interface web ou WhatsApp

Este back-end pode ser integrado a interfaces como:

- Front-end em **Gradio** ou **Streamlit**
- API para **WhatsApp** com **Twilio** ou **Venom Bot**
- Aplicações móveis via **Flutter**, **React Native** ou **Ionic**

---

## 🧠 Modelo utilizado

O modelo usado é configurável via Hugging Face. Exemplo:

```python
model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
```

Você pode substituir por outros modelos compatíveis com quantização, como:

- `meta-llama/Llama-2-7b-chat-hf`
- `mistralai/Mistral-7B-Instruct-v0.2`
- `tiiuae/falcon-7b-instruct`

---

## 🛠️ Dicas para uso eficiente

- Use modelos quantizados (`4bit`, `8bit`) para máquinas com pouca memória.
- Use `device_map={"": "cpu"}` se não tiver GPU ou quiser evitar erros.
- Certifique-se de que `accelerate` esteja instalado e atualizado.
