from pathlib import Path

from ltm_sampling.config import load_config


def test_quickstart_config_has_paired_baseline():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs" / "quickstart.toml")

    assert config.run_count == 6
    assert sum(sampler.method == "full" for sampler in config.samplers) == 1
    assert {dataset.problem_type for dataset in config.datasets} == {
        "classification",
        "regression",
    }
