import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from pymongo.database import Database
from ..database import get_db
from ..models import doc_to_dict
from ..schemas import UserCreate, UserLogin, Token, UserOut
from ..auth import hash_password, verify_password, create_access_token

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(user_in: UserCreate, db: Database = Depends(get_db)):
    # Check if user already exists
    existing_user = db["users"].find_one({"email": user_in.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email already exists."
        )

    # Hash the password and build the user document
    hashed_pwd = hash_password(user_in.password)
    new_user_doc = {
        "name": user_in.name,
        "email": user_in.email,
        "phone": user_in.phone,
        "password_hash": hashed_pwd,
        "role": user_in.role,
        "created_at": datetime.datetime.utcnow()
    }

    result = db["users"].insert_one(new_user_doc)
    created_user = db["users"].find_one({"_id": result.inserted_id})
    return doc_to_dict(created_user)


@router.post("/login", response_model=Token)
def login(credentials: UserLogin, db: Database = Depends(get_db)):
    user_doc = db["users"].find_one({"email": credentials.email})
    if not user_doc or not verify_password(credentials.password, user_doc["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password."
        )

    # Generate token
    access_token = create_access_token(
        data={"sub": user_doc["email"], "role": user_doc["role"]}
    )
    return {"access_token": access_token, "token_type": "bearer"}
