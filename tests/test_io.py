import pytest
import os
import json
from ai_eval_tools.utils.io_utils import read_jsonl, write_jsonl

def test_read_write_jsonl(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    p = d / "test.jsonl"
    
    data = [{"id": 1, "text": "foo"}, {"id": 2, "text": "bar"}]
    write_jsonl(data, p)
    
    assert p.exists()
    
    loaded = read_jsonl(p)
    assert len(loaded) == 2
    assert loaded[0]['text'] == "foo"

def test_malformed_jsonl(tmp_path):
    p = tmp_path / "bad.jsonl"
    with open(p, "w") as f:
        f.write('{"id": 1}\n')
        f.write('BROKEN_JSON\n')
        f.write('{"id": 2}\n')
        
    loaded = read_jsonl(p)
    assert len(loaded) == 2  # Should skip the broken line
