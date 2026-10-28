"""The SSD-MIMO design warnings that tests building the warned configurations on purpose silence, narrowly.

`SSDMimoSSM` warns (RuntimeWarning) when `d_ssm_in`/`d_ssm_out` has exactly one side equal to 1 (the per-subsystem rank
ceiling of Paper 1's crosstalk theorem), and when `b_gate`/`c_gate` is switched off at rank above 1.  A test that is not
about those warnings but constructs such a layer on purpose sets `pytestmark = SILENCE_SSD_DESIGN_WARNINGS` (module) or
decorates the test with it; it ignores exactly these three messages and nothing else.  A test ABOUT a warning asserts it
with `pytest.warns`.  No global filter: pyproject and conftest carry none.
"""
import pytest

#: message regexes (no colon: the filter spec splits at colons; "." matches it) of the three design warnings (src/orbit_ssm/ssm/ssd_mimo.py, `SSDMimoSSM.__init__`)
DESIGN_WARNING_MESSAGES = (
    r"SSDMimoSSM. d_ssm_in=\d+, d_ssm_out=\d+ -- exactly one of them is 1",
    r"SSDMimoSSM\(d_ssm_in=\d+, b_gate=False\). sibling-specific write gating is disabled",
    r"SSDMimoSSM\(d_ssm_out=\d+ > 1, c_gate=False\). output-specific read gating is disabled",
)

SILENCE_SSD_DESIGN_WARNINGS = [pytest.mark.filterwarnings(f"ignore:{m}:RuntimeWarning") for m in DESIGN_WARNING_MESSAGES]
