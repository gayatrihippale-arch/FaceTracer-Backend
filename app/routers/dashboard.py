import datetime
from fastapi import APIRouter, Depends
from pymongo.database import Database
from ..database import get_db
from ..models import doc_to_dict, str_to_objectid
from ..schemas import DashboardStats, SearchHistoryOut
from ..auth import get_current_user

router = APIRouter(
    prefix="/dashboard",
    tags=["Dashboard"]
)


def _build_search_history_out(search_doc: dict, db: Database) -> dict:
    """Embed matched_person into a search history document."""
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


@router.get("/stats", response_model=DashboardStats)
def get_dashboard_stats(
    current_user: dict = Depends(get_current_user),
    db: Database = Depends(get_db)
):
    # Total missing persons registered
    total_missing = db["missing_persons"].count_documents({})

    # Active investigation cases (Status is Active)
    active_cases = db["missing_persons"].count_documents({"status": "Active"})

    # Total searches count
    searches_count = db["search_histories"].count_documents({})

    # Fetch recent searches (top 10), newest first
    recent_search_docs = list(
        db["search_histories"].find({}).sort("search_date", -1).limit(10)
    )
    recent_searches = [_build_search_history_out(doc, db) for doc in recent_search_docs]

    # Generate cases_over_time for last 6 months
    cases_over_time = []
    today = datetime.date.today()

    for i in range(5, -1, -1):
        year = today.year
        month = today.month - i
        if month <= 0:
            month += 12
            year -= 1

        month_start = datetime.datetime(year, month, 1)
        if month == 12:
            next_month_start = datetime.datetime(year + 1, 1, 1)
        else:
            next_month_start = datetime.datetime(year, month + 1, 1)

        month_name = month_start.strftime("%b")

        # Count active & resolved cases created in this month
        active_cnt = db["missing_persons"].count_documents({
            "created_at": {"$gte": month_start, "$lt": next_month_start},
            "status": "Active"
        })
        resolved_cnt = db["missing_persons"].count_documents({
            "created_at": {"$gte": month_start, "$lt": next_month_start},
            "status": "Resolved"
        })

        cases_over_time.append({
            "name": f"{month_name} {year}",
            "Active": active_cnt,
            "Resolved": resolved_cnt
        })

    return {
        "total_missing_persons": total_missing,
        "total_investigation_cases": active_cases,
        "recent_searches_count": searches_count,
        "recent_searches": recent_searches,
        "cases_over_time": cases_over_time
    }
