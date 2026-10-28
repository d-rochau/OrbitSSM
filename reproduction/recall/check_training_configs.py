from pathlib import Path
from experiments.mqmkvar.campaign import get_campaign

root = Path('recall_runs').resolve()
for name in ('p1final-ssd-20260927', 'p1final-hm-20260927'):
    campaign = get_campaign(name)
    for arm in campaign.arms:
        spec = campaign.build_spec('easy', arm, 101, root, smoke=True)
        print(spec.name, spec.train.max_steps)
