from fastapi import FastAPI, Response
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from api import router as api_router
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLaMA Professor API",
    description="API que simula um professor de inglês com LLaMA.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)
app.include_router(api_router)

# Métricas Prometheus
REQUEST_COUNT = Counter("api_request_count", "Número de requisições à API")

@app.middleware("http")
async def metrics_middleware(request, call_next):
    REQUEST_COUNT.inc()
    response = await call_next(request)
    return response

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/")
def root():
    return {"message": "API online. Acesse /docs para usar."}
