
import json
import math

class HMCampaign:

    def __init__(self, path):
        self.data = json.loads(path.read_text())
        self.shapes = self.data['shapes']
        self.time = {}
        for r in self.data['timings']:
            role = '_'.join(r['unit'].split('_')[:2])
            k = (role, r['arm'], r['shape'], r['geometry']['N'], r['pass_'])
            assert k not in self.time
            self.time[k] = r
        self.memory = {(r['level'], r['arm'], r['shape'], r['N']): r for r in self.data['memory']}
        assert len(self.memory) == 100
        self.cost_arms = {'HM m1exact (P=1)': 'HMM:m1exact', 'Mamba 1 (official)': 'official_m1'}
        for p in (32, 64):
            for n in (0, 75, 100):
                self.cost_arms[f'HM-{n}, P={p}'] = f'HM-{n}' + ('@P32' if p == 32 else '')

    def timing(self, role, arm, shape, n, pass_='fwd_bwd'):
        return self.time[role, arm, shape, n, pass_]

    def mem(self, level, arm, shape, n):
        if level == 'mixer' and arm == 'official_m1':
            arm = 'official_mamba1'
        return self.memory[level, arm, shape, n]

    @staticmethod
    def ms(x):
        return f'{x:.2f}' if x < 100 else f'{x:.1f}'

    def overlay_rows(self, name, rows):
        timing = {'table2_main_kernel_fb_ms.tex': ('cost_kernel', 'fwd_bwd', False), 'app_scope_fwd_ms.tex': ('cost_kernel', 'fwd', False), 'app_mixer_fwd_ms.tex': ('cost_mixer', 'fwd', False), 'app_mixer_eager_fb_ms.tex': ('cost_mixer', 'fwd_bwd', False), 'app_mixer_graph_fb_ms.tex': ('cost_mixer', 'fwd_bwd', True)}
        mem = {'app_saved_scope_mib.tex': ('scope', 'candidate_extra_saved_bytes'), 'app_peak_scope_mib.tex': ('scope', 'candidate_peak'), 'app_saved_mixer_mib.tex': ('mixer', 'candidate_extra_saved_bytes'), 'app_peak_mixer_mib.tex': ('mixer', 'candidate_peak')}
        if name in timing or name in mem:
            for row in rows:
                if row[0] not in self.cost_arms:
                    continue
                arm = self.cost_arms[row[0]]
                if name in timing:
                    role, pass_, graph = timing[name]
                    ta = 'official_mamba1' if role == 'cost_mixer' and arm == 'official_m1' else arm
                    if graph:
                        ta += '@graph'
                    row[1:6] = [self.ms(self.timing(role, ta, s, 128, pass_)['candidate_ms']) for s in self.shapes]
                    if name == 'table2_main_kernel_fb_ms.tex':
                        row[6] = f"{self.mem('scope', arm, self.shapes[-1], 128)['candidate_extra_saved_bytes'] / 2 ** 20:.1f}"
                else:
                    level, field = mem[name]
                    row[1:6] = [f'{self.mem(level, arm, s, 128)[field] / 2 ** 20:.0f}' for s in self.shapes]
        elif name == 'table1a_m1.tex':
            rows[0] = ['Large'] + self.m1_summary(self.shapes[:4])
            rows[1][0] += '\\textsuperscript{a}'
        elif name == 'table1a_m1_rest.tex':
            for row in rows:
                row[0] += '\\textsuperscript{a}'
        return rows

    def ratio(self, role, shape, pass_='fwd_bwd', graph=False):
        ours = 'HMM:m1exact' + ('@graph' if graph else '')
        off = ('official_m1' if role == 'm1_kernel' else 'official_mamba1') + ('@graph' if graph else '')
        a = self.timing(role, ours, shape, 16, pass_)['candidate_ms']
        b = self.timing(role, off, shape, 16, pass_)['candidate_ms']
        return (a / b, a, b)

    def m1_summary(self, shapes):
        groups = [[self.ratio(role, s, pass_, graph)[0] for s in shapes] for role, pass_, graph in [('m1_kernel', 'fwd', False), ('m1_kernel', 'fwd_bwd', False), ('m1_mixer', 'fwd_bwd', False), ('m1_mixer', 'fwd_bwd', True)]]
        for field in ['candidate_extra_saved_bytes', 'candidate_peak']:
            groups.append([self.mem('mixer', 'HMM:m1exact', s, 16)[field] / self.mem('mixer', 'official_m1', s, 16)[field] for s in shapes])
        return [f'{math.exp(sum(map(math.log, v)) / len(v)):.2f} ({min(v):.2f}--{max(v):.2f})' for v in groups]

    def recovery(self, recovery):
        for cell in recovery['pairs']['1a_m1']['cells']:
            s = cell['shape']
            for scope in ['kernel', 'mixer']:
                ratio, a, b = self.ratio('m1_' + scope, s)
                cell[scope] = {'ours_ms': a, 'official_ms': b, 'ratio': ratio, 'source': 'hm_optimization.json', 'candidate_sha': self.data['setup']['candidate_sha']}
                a = self.mem('scope' if scope == 'kernel' else 'mixer', 'HMM:m1exact', s, 16)
                b = self.mem('scope' if scope == 'kernel' else 'mixer', 'official_m1', s, 16)
                cell[scope + '_peak'] = {'ours_bytes': a['candidate_peak'], 'official_bytes': b['candidate_peak'], 'ratio': a['candidate_peak'] / b['candidate_peak'], 'sources': [a['candidate_source'], b['candidate_source']]}
        recovery['selection'] = 'Original accepted SSD/Mamba 2/3 pairs; audited five-shape HM campaign replaces both arms of the Mamba 1 pair. All ratios use unrounded absolute values.'
        recovery['hm_update'] = {'source': 'hm_optimization.json', 'candidate_sha': self.data['setup']['candidate_sha'], 'scope': 'M1 pair only; both sides from same campaign. Other pairs unchanged.'}
        return recovery

    def validate_retained(self, memory):
        for group in ['table3_kernel', 'table3_mixer', 'table4_scope']:
            level = 'mixer' if group == 'table3_mixer' else 'scope'
            for r in memory[group]:
                if not r['arm'].startswith(('HM', 'official_m1', 'official_mamba1')):
                    continue
                s = self.shapes[-1] if r['shape_tag'] == 'lm' else r['shape_tag']
                new = self.mem(level, r['arm'], s, r['geometry']['N'])
                assert r['total_retained_nonparameter_bytes'] == new['candidate_retained'], (group, r['arm'], s)

    def detail(self, old_rows):
        result = []
        for s in self.shapes:
            vals = []
            for role, pass_, graph in [('m1_kernel', 'fwd', False), ('m1_kernel', 'fwd_bwd', False), ('m1_mixer', 'fwd_bwd', False), ('m1_mixer', 'fwd_bwd', True)]:
                ratio, a, b = self.ratio(role, s, pass_, graph)
                vals.append(f'{ratio:.2f} ({a:.3f}/{b:.3f})')
            result.append(['\\texttt{' + s.replace('_', '\\_') + '}'] + vals)
        old = [r for r in old_rows if r[0] not in {row[0] for row in result}]
        assert len(old) == 5
        for row in old:
            row[0] += '\\textsuperscript{a}'
        return result + old
