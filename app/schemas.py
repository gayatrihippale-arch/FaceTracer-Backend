from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Any
from datetime import datetime


# --- Token Schemas ---
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: Optional[str] = None
    role: Optional[str] = None


# --- User Schemas ---
class UserBase(BaseModel):
    name: str
    email: EmailStr
    phone: Optional[str] = None
    role: str = "Police"  # Police / Admin


class UserCreate(UserBase):
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserUpdate(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    role: Optional[str] = None
    password: Optional[str] = None


class UserOut(UserBase):
    id: str
    created_at: datetime

    model_config = {"from_attributes": True}


# --- Missing Person Schemas ---
class MissingPersonBase(BaseModel):
    name: str
    age: int
    gender: str
    last_seen_location: str
    last_seen_date: datetime
    description: Optional[str] = None
    status: str = "Active"


class MissingPersonCreate(MissingPersonBase):
    pass


class MissingPersonOut(MissingPersonBase):
    id: str
    photo_path: str
    created_at: datetime
    created_by: str  # Now a string ObjectId reference

    model_config = {"from_attributes": True}


class MissingPersonUpdate(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    last_seen_location: Optional[str] = None
    last_seen_date: Optional[datetime] = None
    description: Optional[str] = None
    status: Optional[str] = None


# --- Search History Schemas ---
class SearchHistoryOut(BaseModel):
    id: str
    searched_photo_path: str
    search_date: datetime
    searched_by: str  # String ObjectId reference
    match_found: bool
    matched_person_id: Optional[str] = None
    confidence_score: Optional[float] = None
    matched_person: Optional[MissingPersonOut] = None

    model_config = {"from_attributes": True}


# --- Dashboard Stats Schema ---
class DashboardStats(BaseModel):
    total_missing_persons: int
    total_investigation_cases: int  # Total active missing persons
    recent_searches_count: int
    recent_searches: List[SearchHistoryOut]
    cases_over_time: List[dict]  # Data points for Recharts charts
