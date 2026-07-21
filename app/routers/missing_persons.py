import os
import uuid
import json
import datetime
import re
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from pymongo.database import Database
from ..database import get_db
from ..models import doc_to_dict, str_to_objectid
from ..schemas import MissingPersonOut
from ..auth import get_current_user
from ..services.face_recognition_service import face_recognition_service

router = APIRouter(
    prefix="/missing-persons",
    tags=["Missing Persons"]
)

# Root directory for uploads (accessible as static files)
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("", response_model=MissingPersonOut, status_code=status.HTTP_201_CREATED)
def register_missing_person(
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    last_seen_location: str = Form(...),
    last_seen_date: str = Form(...),
    description: Optional[str] = Form(None),
    status_str: str = Form("Active", alias="status"),
    photo: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db: Database = Depends(get_db)
):
    # Create unique filename
    file_extension = os.path.splitext(photo.filename)[1]
    if not file_extension:
        file_extension = ".jpg"
    unique_filename = f"{uuid.uuid4().hex}{file_extension}"
    photo_path = os.path.join(UPLOAD_DIR, unique_filename)

    # Save uploaded file
    try:
        with open(photo_path, "wb") as f:
            f.write(photo.file.read())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save image: {str(e)}"
        )

    # Relative path for frontend static access
    relative_photo_path = f"uploads/{unique_filename}"

    # Parse date
    try:
        parsed_date = datetime.datetime.fromisoformat(last_seen_date.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed_date = datetime.datetime.strptime(last_seen_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid date format. Use ISO format or YYYY-MM-DD."
            )

    # Extract Face Encoding
    face_encoding = face_recognition_service.get_face_encoding(photo_path)
    serialized_encoding = json.dumps(face_encoding) if face_encoding is not None else None

    # Build and insert MongoDB document
    new_person_doc = {
        "name": name,
        "age": age,
        "gender": gender,
        "last_seen_location": last_seen_location,
        "last_seen_date": parsed_date,
        "description": description,
        "photo_path": relative_photo_path,
        "face_encoding": serialized_encoding,
        "status": status_str,
        "created_at": datetime.datetime.utcnow(),
        "created_by": current_user["id"]  # Store as string reference
    }

    result = db["missing_persons"].insert_one(new_person_doc)
    created = db["missing_persons"].find_one({"_id": result.inserted_id})
    return doc_to_dict(created)


@router.get("", response_model=list[MissingPersonOut])
def list_missing_persons(
    search: Optional[str] = None,
    gender: Optional[str] = None,
    status: Optional[str] = None,
    age_min: Optional[int] = None,
    age_max: Optional[int] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    db: Database = Depends(get_db)
):
    query_filter = {}

    # Text search on name or location using case-insensitive regex
    if search:
        pattern = re.compile(re.escape(search), re.IGNORECASE)
        query_filter["$or"] = [
            {"name": {"$regex": pattern}},
            {"last_seen_location": {"$regex": pattern}}
        ]

    # Category filters
    if gender:
        query_filter["gender"] = gender
    if status:
        query_filter["status"] = status

    # Age range filters
    if age_min is not None or age_max is not None:
        age_filter = {}
        if age_min is not None:
            age_filter["$gte"] = age_min
        if age_max is not None:
            age_filter["$lte"] = age_max
        query_filter["age"] = age_filter

    # Date range filters
    if date_from or date_to:
        date_filter = {}
        if date_from:
            try:
                df = datetime.datetime.fromisoformat(date_from.replace("Z", "+00:00"))
                date_filter["$gte"] = df
            except ValueError:
                pass
        if date_to:
            try:
                dt = datetime.datetime.fromisoformat(date_to.replace("Z", "+00:00"))
                date_filter["$lte"] = dt
            except ValueError:
                pass
        if date_filter:
            query_filter["last_seen_date"] = date_filter

    persons = list(
        db["missing_persons"].find(query_filter).sort("created_at", -1)
    )
    return [doc_to_dict(p) for p in persons]


@router.get("/{id}", response_model=MissingPersonOut)
def get_missing_person(
    id: str,
    current_user: dict = Depends(get_current_user),
    db: Database = Depends(get_db)
):
    try:
        oid = str_to_objectid(id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid ID format.")

    person = db["missing_persons"].find_one({"_id": oid})
    if not person:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Missing person record not found."
        )
    return doc_to_dict(person)


@router.put("/{id}", response_model=MissingPersonOut)
def update_missing_person(
    id: str,
    name: Optional[str] = Form(None),
    age: Optional[int] = Form(None),
    gender: Optional[str] = Form(None),
    last_seen_location: Optional[str] = Form(None),
    last_seen_date: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    status: Optional[str] = Form(None),
    photo: Optional[UploadFile] = File(None),
    current_user: dict = Depends(get_current_user),
    db: Database = Depends(get_db)
):
    try:
        oid = str_to_objectid(id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid ID format.")

    person = db["missing_persons"].find_one({"_id": oid})
    if not person:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Missing person record not found."
        )

    update_fields = {}

    if name is not None:
        update_fields["name"] = name
    if age is not None:
        update_fields["age"] = age
    if gender is not None:
        update_fields["gender"] = gender
    if last_seen_location is not None:
        update_fields["last_seen_location"] = last_seen_location
    if description is not None:
        update_fields["description"] = description
    if status is not None:
        update_fields["status"] = status

    if last_seen_date is not None:
        try:
            update_fields["last_seen_date"] = datetime.datetime.fromisoformat(
                last_seen_date.replace("Z", "+00:00")
            )
        except ValueError:
            pass

    if photo is not None:
        file_extension = os.path.splitext(photo.filename)[1] or ".jpg"
        unique_filename = f"{uuid.uuid4().hex}{file_extension}"
        photo_path = os.path.join(UPLOAD_DIR, unique_filename)
        with open(photo_path, "wb") as f:
            f.write(photo.file.read())
        update_fields["photo_path"] = f"uploads/{unique_filename}"

        # Recalculate face encoding
        face_encoding = face_recognition_service.get_face_encoding(photo_path)
        update_fields["face_encoding"] = json.dumps(face_encoding) if face_encoding is not None else None

    if update_fields:
        db["missing_persons"].update_one({"_id": oid}, {"$set": update_fields})

    updated = db["missing_persons"].find_one({"_id": oid})
    return doc_to_dict(updated)


@router.delete("/{id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_missing_person(
    id: str,
    current_user: dict = Depends(get_current_user),
    db: Database = Depends(get_db)
):
    try:
        oid = str_to_objectid(id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid ID format.")

    result = db["missing_persons"].delete_one({"_id": oid})
    if result.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Missing person record not found."
        )
