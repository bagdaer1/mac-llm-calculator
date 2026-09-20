"""The model has to reproduce the runs it was fitted on."""

from maccalc import MOE_READ_FACTOR, data, fit_model, tokens_per_second, weights_gb

BENCH = data.benchmarks()
BANDWIDTH = {"m4-base": 120.0, "m5-pro": 307.0}
Q4_BPW = 4.85
TOLERANCE = 0.047
"""Worst case in the fit is 4.63%, on DeepSeek R1 8B. Leave a float hair above."""


def _dense_runs():
    return [r for r in BENCH["runs"] if not r.get("active_b")]


def test_reproduces_every_dense_run_within_tolerance():
    worst = 0.0
    for run in _dense_runs():
        read = weights_gb(run["params_b"], Q4_BPW)
        predicted = tokens_per_second(BANDWIDTH[run["machine"]], read)
        error = abs(predicted - run["gen_tok_s"]) / run["gen_tok_s"]
        worst = max(worst, error)
        assert error <= TOLERANCE, (
            f"{run['model']} on {run['machine']}: predicted {predicted}, measured {run['gen_tok_s']}"
        )
    assert worst > 0.0


def test_covers_two_machines():
    machines = {r["machine"] for r in BENCH["runs"]}
    assert machines == {"m4-base", "m5-pro"}, machines


def test_moe_runs_match_the_read_factor():
    """The correction exists because the active weights alone predict too fast."""
    for run in BENCH["runs"]:
        if not run.get("active_b"):
            continue
        read = weights_gb(run["active_b"], Q4_BPW) * MOE_READ_FACTOR
        predicted = tokens_per_second(BANDWIDTH[run["machine"]], read)
        error = abs(predicted - run["gen_tok_s"]) / run["gen_tok_s"]
        assert error <= 0.05, f"{run['model']}: predicted {predicted}, measured {run['gen_tok_s']}"


def test_moe_speed_uses_active_parameters():
    dense = fit_model(params_b=671, bits_per_weight=Q4_BPW, memory_gb=512, bandwidth_gbs=819, ctx_overhead_gb=10)
    moe = fit_model(params_b=671, bits_per_weight=Q4_BPW, memory_gb=512, bandwidth_gbs=819, ctx_overhead_gb=10, active_params_b=37)
    assert moe.weights_gb == dense.weights_gb
    assert moe.tokens_per_second > dense.tokens_per_second


def test_a_model_that_does_not_fit_has_no_speed():
    fit = fit_model(params_b=70.6, bits_per_weight=Q4_BPW, memory_gb=16, bandwidth_gbs=120, ctx_overhead_gb=3)
    assert not fit.fits and fit.tokens_per_second is None


def test_tables_load():
    assert len(data.chips()) >= 20
    assert len(data.accelerators()) >= 6
    assert any(m["slug"] == "llama-3-3-70b" for m in data.models())
