import os
import uuid
import json
import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from pymongo.database import Database
from ..database import get_db
from ..models import doc_to_dict, str_to_objectid
from ..schemas import SearchHistoryOut
from ..auth import get_current_user
from ..services.face_recognition_service import face_recognition_service

router = APIRouter(
    prefix="/face-search",
    tags=["Face Search"]
)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")
SEARCHES_DIR = os.path.join(UPLOAD_DIR, "searches")
os.makedirs(SEARCHES_DIR, exist_ok=True)


def _build_search_history_out(search_doc: dict, db: Database) -> dict:
    """
    Convert a search_history document to a serializable dict,
    and optionally embed the matched_person sub-document.
    """
    result = doc_to_dict(search_doc)

    matched_person_id = result.get("matched_person_id")
    if matched_person_id:
        try:
            person_doc = db["missing_persons"].find_one({"_id": str_to_objectid(matched_person_id)})
            result["matched_person"] = doc_to_dict(person_doc) if person_doc else None
        except Exception:
            result["matched_person"] = None
    else:
        result["matched_person"] = None

    return result


@router.post("", response_model=SearchHistoryOut)
def face_search(
    photo: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db: Database = Depends(get_db)
):
    # Save the query search image
    file_extension = os.path.splitext(photo.filename)[1] or ".jpg"
    unique_filename = f"{uuid.uuid4().hex}{file_extension}"
    saved_path = os.path.join(SEARCHES_DIR, unique_filename)

    try:
        with open(saved_path, "wb") as f:
            f.write(photo.file.read())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save query image: {str(e)}"
        )

    relative_photo_path = f"uploads/searches/{unique_filename}"

    # Extract Face Encoding for the query image
    query_encoding = face_recognition_service.get_face_encoding(saved_path)
    if query_encoding is None:
        # No face detected — save record and return
        new_search_doc = {
            "searched_photo_path": relative_photo_path,
            "search_date": datetime.datetime.utcnow(),
            "searched_by": current_user["id"],
            "match_found": False,
            "matched_person_id": None,
            "confidence_score": 0.0
        }
        result = db["search_histories"].insert_one(new_search_doc)
        saved = db["search_histories"].find_one({"_id": result.inserted_id})
        return _build_search_history_out(saved, db)

    # Fetch all missing persons with valid face encodings that are Active
    missing_persons = list(
        db["missing_persons"].find(
            {"face_encoding": {"$ne": None}, "status": "Active"}
        )
    )

    best_match_person = None
    best_similarity = -2.0
    best_confidence = 0.0

    for person in missing_persons:
        try:
            db_encoding = json.loads(person["face_encoding"])
            if db_encoding is None:
                continue

            similarity, confidence = face_recognition_service.match_faces(query_encoding, db_encoding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_confidence = confidence
                best_match_person = person
        except Exception as e:
            print(f"Error matching with person {person.get('_id')}: {e}")
            continue

    # Threshold checks — confidence >= 70% is a match
    is_match = False
    matched_id = None
    final_confidence = 0.0

    if best_match_person is not None:
        if best_confidence >= 70.0:
            is_match = True
            matched_id = str(best_match_person["_id"])
            final_confidence = best_confidence
        else:
            is_match = False
            matched_id = None
            final_confidence = best_confidence

    # Create Search History entry in MongoDB
    new_search_doc = {
        "searched_photo_path": relative_photo_path,
        "search_date": datetime.datetime.utcnow(),
        "searched_by": current_user["id"],
        "match_found": is_match,
        "matched_person_id": matched_id,
        "confidence_score": final_confidence if is_match else best_confidence
    }

    result = db["search_histories"].insert_one(new_search_doc)
    saved = db["search_histories"].find_one({"_id": result.inserted_id})
    return _build_search_history_out(saved, db)
