from cobaya.model import get_model
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt



def fake_likelihood(
        _theory={'dndm': None, 'power': None, "MF": None}):
    dndm = _theory.get_result('dndm')
    print(dndm[0])
    return stats.norm.logpdf(0, loc=0, scale=0.1)


fiducial_params = {
    "omegabh2": 0.02225,
    "omegach2": 0.1198,
    "H0": 67.3,
    "tau": 0.06,
    "As": 2.2e-9,
    "ns": 0.96,
}

zs = np.linspace(0, 2, 40)
info = {
    "debug": True,
    "params": fiducial_params,
    "likelihood": {
        'test_likelihood': fake_likelihood
    },
    "theory": {
        "classy": {
            "extra_args": {}
        },
        "cobayahacks.theories.hmf": {
            "zarr": zs,
            "hmf_kwargs": {}
        }
    },
}


m = get_model(info)

print(m.loglike())

mf = m.provider.get_result("MF")
power = m.provider.get_result("power")
plt.plot(mf.k, power[0])
plt.xscale('log')
plt.yscale('log')
plt.show()

