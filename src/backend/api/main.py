from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="SCAN API",
    description="Backend API for the Soccer Cognitive AI Network",
    version="0.1.0",
)


# Allow the local frontend to communicate with FastAPI.
# Next.js normally runs on localhost:3000 during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "name": "SCAN",
        "full_name": "Soccer Cognitive AI Network",
        "status": "online",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "SCAN API",
    }