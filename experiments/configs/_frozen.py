
from __future__ import annotations
FROZEN_MANIFESTS = {'P1_CONFIG_PRIORITIES_20260906.json': 'MQMKVAR P1 grid 2026-09-11..14 (experiments/configs/mqmkvar_p1.py; results redacted/mqmkvar-p1/results)', 'HM_GATES_INSANE_MANIFEST_20260922.json': "HM gate screen 2026-09-22 (real arms complete; rot arms cancelled by the owner's real-only scope; redacted/hm-gates-insane-20260922)", 'P1_MQMKVAR_MATCHING_CANDIDATE_20260920.json': 'matched MQMKVAR baselines 2026-09-20 (its ssd_mimo rows were never built by a factory; the official-baseline arms stay runnable)', 'P1_SSD_ABLATION_MATCHED_20260920.json': 'real R2 insane factorial 2026-09-20 (PAPER1_MQMKVAR_RESULTS_20260920.md)'}

class FrozenManifestError(RuntimeError):
    pass

def refuse(manifest: str, what: str) -> None:
    raise FrozenManifestError(f'{what}: implementationexperiments/{manifest} is FROZEN provenance ({FROZEN_MANIFESTS[manifest]}).  It spells SSD-MIMO/HybridMamba in the pre-spec vocabulary (rot_mode/a_dyn/rope_fraction/fracs/gates/bc_bias), which this main refuses without migration (owner 2026-09-24).  A re-run needs a new dated manifest spelled with rows/rotation/maps (implementation.md implementation.3).')
