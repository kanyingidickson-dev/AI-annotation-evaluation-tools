from pydantic import BaseModel
from typing import Optional

class AnnotationEntry(BaseModel):
    """
    Standard schema for a single data entry in the pipeline.
    """
    id: int
    text: str
    label: Optional[str] = None
    score: Optional[float] = None
    feedback: Optional[str] = None
