import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger("tutor")

CONTENT_PATH = Path(__file__).parent.parent / "shared-data" / "day4_tutor_content.json"


def load_tutor_content() -> List[Dict]:
    """Load list of concepts from the shared JSON file."""
    if not CONTENT_PATH.exists():
        logger.warning(f"Tutor content file not found at {CONTENT_PATH}")
        return []

    try:
        with CONTENT_PATH.open("r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read tutor content: {e}")
        return []


def get_concept_by_id(content: List[Dict], concept_id: str) -> Optional[Dict]:
    for c in content:
        if c.get("id") == concept_id:
            return c
    return None


def list_concept_ids_and_titles(content: List[Dict]) -> str:
    if not content:
        return "No concepts available."
    return ", ".join(f"{c['id']} ({c['title']})" for c in content)
