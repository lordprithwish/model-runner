from fastapi import FastAPI

app = FastAPI(title="Model Runner")

@app.get("/health")
def health():
    return {"status": "ok"}