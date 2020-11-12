"""Test HMM found in model directories."""

import pytest
import os
from pathlib import Path
from importlib import import_module
import warnings

import dapper as dpr
import dapper.tools.utils as utils
utils.disable_progbar = True

modules_with_HMM = []

for root, dir, files in os.walk("."):
    if "mods" in root:

        # Can uncomment if you have compiled and generated samples
        # if "QG" in root:
        #     continue

        for f in sorted(files):
            if f.endswith(".py"):
                filepath = Path(root) / f

                lines = "".join(open(filepath).readlines())
                if "HiddenMarkovModel" in lines:
                    modules_with_HMM.append(filepath)



@pytest.mark.parametrize(("path"), modules_with_HMM)
def test_HMM(path):
    """Test that any HMM in module can be simulated."""
    p = str(path.with_suffix("")).replace("/", ".")
    module = import_module(p)

    def exclude(key, HMM):
        """Exclude HMMs that are not testable w/o further configuration."""
        if key == "HMM_trunc":
            return True
        return False

    for key, HMM in vars(module).items():
        if isinstance(HMM, dpr.HiddenMarkovModel) and not exclude(key,HMM):
            HMM.t.BurnIn = 0
            HMM.t.KObs = 1
            xx, yy = HMM.simulate()
            assert True