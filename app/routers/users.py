from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, status
from pymongo.database import Database
from ..database import get_db
from ..models import doc_to_dict, str_to_objectid
from ..schemas import UserOut, UserUpdate
from ..auth import get_current_user, hash_password

router = APIRouter(
    prefix="/users",
    tags=["Users"]
)


@router.get("/me", response_model=UserOut)
def get_me(current_user: dict = Depends(get_current_user)):
    return current_user


@router.put("/me", response_model=UserOut)
def update_profile(
    profile_data: UserUpdate,
    current_user: dict = Depends(get_current_user),
    db: Database = Depends(get_db)
):
    update_fields = {}

    if profile_data.name is not None:
        update_fields["name"] = profile_data.name
    if profile_data.phone is not None:
        update_fields["phone"] = profile_data.phone
    if profile_data.role is not None:
        update_fields["role"] = profile_data.role
    if profile_data.password is not None and profile_data.password != "":
        update_fields["password_hash"] = hash_password(profile_data.password)

    if update_fields:
        db["users"].update_one(
            {"_id": str_to_objectid(current_user["id"])},
            {"$set": update_fields}
        )

    updated_user = db["users"].find_one({"_id": str_to_objectid(current_user["id"])})
    return doc_to_dict(updated_user)
