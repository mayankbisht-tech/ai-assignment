import os
import sys
import glob
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from eval import evaluate_file
from app import load_data


def test_eval_semantics_on_outputs():
    data = load_data()
    all_ids = {item['id'] for item in data['suppliers']}
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    files = sorted(glob.glob(os.path.join(out_dir, '*.json')))
    assert files, "No outputs to evaluate"
    # run evaluate_file on all outputs and assert no problems
    for f in files:
        probs = evaluate_file(f, all_ids, data)
        assert not probs, f"Eval found problems in {f}: {probs}"
