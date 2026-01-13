import pytest
from ai_eval_tools.core.llm_client import LLMClient
from ai_eval_tools.scripts.pipeline_autonomous import AnnotationEntry

def test_annotation_entry_validation():
    """Test that pydantic models validate data correctly."""
    
    # Valid
    entry = AnnotationEntry(id=1, text="Test", label="foo", score=0.9)
    assert entry.id == 1
    assert entry.score == 0.9
    
    # Invalid type
    with pytest.raises(ValueError):
        AnnotationEntry(id="not-an-int", text="Test")

def test_llm_client_init_error():
    """Test that LLM client raises error for unsupported models."""
    with pytest.raises(ValueError):
        LLMClient(model="unsupported-model-9000", api_key="fake")
