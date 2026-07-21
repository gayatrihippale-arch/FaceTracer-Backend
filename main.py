import os
from dotenv import load_dotenv

# Load environment variables from .env BEFORE any app imports
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routers import auth, users, missing_persons, face_search, dashboard

app = FastAPI(
    title="FaceTracer API",
    description="Backend API for Face Recognition Missing Person Finder System",
    version="1.0.0"
)

# CORS configuration — allow all origins for easy deployment
# In production, replace "*" with your Vercel frontend domain
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount uploads directory to serve images statically
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Include Routers
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(missing_persons.router)
app.include_router(face_search.router)
app.include_router(dashboard.router)


@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Welcome to FaceTracer API. System is fully operational.",
        "version": "1.0.0",
        "database": "MongoDB Atlas"
    }


@app.on_event("startup")
async def startup_db_check():
    """Verify MongoDB connection on startup."""
    try:
        from app.database import client
        # Ping the deployment to confirm connection
        client.admin.command("ping")
        print("[OK] Successfully connected to MongoDB Atlas!")
    except Exception as e:
        print(f"[ERROR] Failed to connect to MongoDB Atlas: {e}")
        raise RuntimeError(f"MongoDB connection failed: {e}")
