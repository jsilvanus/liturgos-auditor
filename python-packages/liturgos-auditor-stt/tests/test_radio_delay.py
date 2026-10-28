from auditor_stt.serve.live.radio_delay import AdaptiveDelay


def test_delay_starts_at_minimum_and_backs_off_on_caption_lateness():
    policy = AdaptiveDelay(minimum_seconds=0, maximum_seconds=8, initial_step_seconds=0.5)
    assert policy.snapshot()["target_delay_seconds"] == 0
    first = policy.report(3000)
    assert first["target_delay_seconds"] == 0.5
    assert first["reason"] == "caption_late"
    second = policy.report(3000)
    assert second["target_delay_seconds"] == 1.5
    assert second["reason"] == "caption_late"


def test_delay_is_bounded_at_maximum():
    policy = AdaptiveDelay(minimum_seconds=0, maximum_seconds=1, initial_step_seconds=0.5)
    policy.report(10000)
    assert policy.report(10000)["target_delay_seconds"] == 1


def test_delay_only_steps_down_after_sustained_stability():
    policy = AdaptiveDelay(
        minimum_seconds=0,
        maximum_seconds=5,
        initial_step_seconds=0.5,
        stable_reports_to_decrease=3,
        decrease_step_seconds=0.25,
    )
    policy.report(2000)
    assert policy.snapshot()["target_delay_seconds"] == 0.5
    policy.report(0)
    policy.report(0)
    assert policy.snapshot()["target_delay_seconds"] == 0.5
    result = policy.report(0)
    assert result["target_delay_seconds"] == 0.25
    assert result["reason"] == "stable_step_down"


def test_early_captions_do_not_increase_delay():
    policy = AdaptiveDelay(minimum_seconds=0, maximum_seconds=5)
    result = policy.report(-2000)
    assert result["target_delay_seconds"] == 0
    assert result["reason"] == "caption_early"


def test_explicit_delay_request_is_bounded():
    policy = AdaptiveDelay(minimum_seconds=0, maximum_seconds=5)
    assert policy.request(2.5)["target_delay_seconds"] == 2.5
    try:
        policy.request(6)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range delay must fail")
