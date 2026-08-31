from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import inference

app = FastAPI(title="Liver Cancer Retrieval API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(inference.router, prefix="/api")
