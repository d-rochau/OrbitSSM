
import os
from triton.runtime.cache import FileCacheManager
from triton import knobs

class LayeredCacheManager(FileCacheManager):

    def __init__(self, key, override=False, dump=False):
        super().__init__(key, override=override, dump=dump)
        ro = os.environ.get('PB_RO_CACHE', '')
        self.ro_dir = os.path.join(ro, key) if ro and (not override) and (not dump) else None

    def get_file(self, filename):
        p = super().get_file(filename)
        if p is None and self.ro_dir:
            q = os.path.join(self.ro_dir, filename)
            if os.path.exists(q):
                return q
        return p

    def get_group(self, filename):
        g = super().get_group(filename)
        if g is None and self.ro_dir:
            gp = os.path.join(self.ro_dir, f'__grp__{filename}')
            if os.path.exists(gp):
                import json
                try:
                    cp = json.load(open(gp)).get('child_paths')
                except Exception:
                    return None
                if cp and all((os.path.exists(p) for p in cp.values())):
                    return dict(cp)
        return g
