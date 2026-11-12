import importlib.util
from pathlib import Path


def test_verification_example_runs_and_validates_its_claims():
    path = Path(__file__).parents[1] / "examples" / "verify_mechanisms.py"
    spec = importlib.util.spec_from_file_location("verify_mechanisms", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.verify()
    assert result["attention_max_abs_error"] < 1e-10
    assert result["smoke"]["final_loss"] < result["smoke"]["initial_loss"]
