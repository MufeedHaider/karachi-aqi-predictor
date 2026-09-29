"""The analyst must never pass on a number no tool produced.

Runs with a scripted stand-in for the LLM: no API key, no network.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import analyst  # noqa: E402
from analyst import ask, get_forecast, ungrounded_numbers  # noqa: E402
from run_analyst_evals import judge, load_cases  # noqa: E402


class ScriptedLLM:
    """Calls the named tools, then returns canned answers in order."""

    def __init__(self, tool_names, answers):
        self.tool_names, self.answers, self.calls = tool_names, list(answers), 0

    def run(self, question, tools):
        by_name = {t.__name__: t for t in tools}
        for name in self.tool_names:
            by_name[name]()
        self.calls += 1
        return self.answers.pop(0)


def _peak():
    return get_forecast()["peak"]


def test_tools_read_the_committed_forecast():
    out = get_forecast(1, 72)
    assert "error" not in out
    assert len(out["hours"]) == 72
    assert out["peak"]["pm2_5"] >= out["mean_pm2_5"] >= out["lowest"]["pm2_5"]


def test_grounded_answer_passes_first_time():
    p = _peak()
    llm = ScriptedLLM(["get_forecast"],
                      [f"Worst hour is {p['time']} at {p['pm2_5']} µg/m³, AQI {p['aqi']}."])
    a = ask("when is it worst?", llm=llm)
    assert a.grounded and a.attempts == 1 and a.tools_called == ["get_forecast"]


def test_invented_number_triggers_one_regeneration():
    p = _peak()
    llm = ScriptedLLM(["get_forecast"],
                      ["PM2.5 will reach 187.3 tomorrow.",
                       f"The peak is {p['pm2_5']} µg/m³ at {p['time']}."])
    a = ask("how bad tomorrow?", llm=llm)
    assert a.attempts == 2 and a.grounded and "187.3" not in a.text


def test_persistent_invention_is_never_shown():
    llm = ScriptedLLM(["get_forecast"], ["It will be 187.3.", "Definitely 187.3."])
    a = ask("how bad tomorrow?", llm=llm)
    assert not a.grounded
    assert "187.3" not in a.text.split("here is the data itself")[0]


def test_percent_form_of_a_fraction_counts_as_grounded():
    assert ungrounded_numbers("It beats CAMS by 45.7%.", [{"skill_vs_cams": 0.457}]) == []


def test_rounding_is_tolerated_but_not_new_values():
    outputs = [{"pm2_5": 34.62}]
    assert ungrounded_numbers("about 34.6", outputs) == []
    assert ungrounded_numbers("about 35", outputs) == []
    assert ungrounded_numbers("about 41.0", outputs) == ["41.0"]


def test_small_counts_and_lead_times_are_not_claims():
    assert ungrounded_numbers("in the next 72 hours, 3 days, 80% range", []) == []


def test_eval_set_is_well_formed():
    cases = load_cases()
    names = {t.__name__ for t in analyst.TOOLS}
    assert len(cases) >= 20
    assert len({c["id"] for c in cases}) == len(cases)
    for c in cases:
        assert set(c.get("expect_tools", [])) <= names
        assert c.get("expect_tools") or c.get("expect_any")


def test_out_of_scope_is_judged_on_the_decline():
    case = {"id": "x", "question": "next month?", "expect_tools": [],
            "expect_any": ["only", "72"]}
    a = analyst.Answer(text="I only forecast the next 72 hours.")
    assert judge(case, a)["right_tool"]
