
from __future__ import annotations
from typing import Callable, Dict, List
from experiments.specs import IntegrationSpec
CONFIG_FACTORIES: Dict[str, Callable[[], List[IntegrationSpec]]] = {}

def register(name: str):

    def deco(fn: Callable[[], List[IntegrationSpec]]):
        if name in CONFIG_FACTORIES:
            raise ValueError(f'duplicate config factory: {name}')
        CONFIG_FACTORIES[name] = fn
        return fn
    return deco
from experiments.configs import mqmkvar_p1 as _mqmkvar_p1
