"""Feature: credit-score time machine."""
from loanlens.timeline import HABITS, first_month, simulate


def test_good_habits_raise_score_and_eventually_approve(shared_predictor, samples):
    sim = simulate(shared_predictor.model, samples["rejected"], ["on_time", "low_use", "no_new"], months=36)
    assert len(sim) == 37 and sim["month"].iloc[0] == 0
    assert sim["cibil_score"].is_monotonic_increasing
    assert sim["cibil_score"].max() <= 900
    assert sim["probability"].iloc[0] < 0.5
    m = first_month(sim, 0.6)
    assert m is not None and sim.loc[sim["month"] == m, "probability"].iloc[0] >= 0.6


def test_missed_payment_lowers_score_and_savings_add_up(shared_predictor, samples):
    a = samples["approved"]
    sim = simulate(shared_predictor.model, a, ["missed"], months=6, monthly_saving=100_000)
    assert sim.loc[sim["month"] == 3, "cibil_score"].iloc[0] == a["cibil_score"] - 70
    assert sim["bank_asset_value"].iloc[-1] == a["bank_asset_value"] + 600_000


def test_point_overrides_and_no_crossing(shared_predictor, samples):
    zero = {h.key: 0 for h in HABITS}
    sim = simulate(shared_predictor.model, samples["rejected"], ["on_time"], months=12, points=zero)
    assert sim["cibil_score"].nunique() == 1
    assert first_month(sim, 0.6) is None
