"""The model has to reproduce the runs it was fitted on."""

from maccalc import data, fit_model, tokens_per_second, weights_gb

BENCH = data.benchmarks()
M4_BANDWIDTH = 120.0
Q4_BPW = 4.85


def test_reproduces_every_measured_run_within_4_percent():
    worst = 0.0
    for run in BENCH["runs"]:
        read = weights_gb(run["params_b"], Q4_BPW)
        predicted = tokens_per_second(M4_BANDWIDTH, read)
        error = abs(predicted - run["gen_tok_s"]) / run["gen_tok_s"]
        worst = max(worst, error)
        # worst case is DeepSeek R1 8B at exactly 4.0%, so allow a float hair above
        assert error <= 0.0401, f"{run['model']}: predicted {predicted}, measured {run['gen_tok_s']}"
    assert worst > 0.0


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
