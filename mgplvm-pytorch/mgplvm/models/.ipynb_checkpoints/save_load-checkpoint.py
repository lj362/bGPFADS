import torch
from . import sgp, svgp
from .. import rdist, kernels, utils
from ..manifolds.base import Manifold
from ..kernels import Kernel


def save_model(mod, **kwargs):
    torch.save(mod.state_dict(), fname + '.pt')

    params = {
        'model': mod.name,
        'manif': mod.manif.name,
        'kernel': mod.kernel.name,
        'rdist': mod.rdist.name,
        'n_z': mod.z.n_z,
        'n': mod.n,
        'm': mod.m,
        'd': mod.d
    }
    for key, value in kwargs.items():
        params[key] = value
    pickle.dump(params, open(fname + '.pickled', 'wb'))


def recover_model(fname, device):
    params = pickle.load(open(fname + '.pickled', 'rb'))
    manifdict = {'Torus': Torus, 'Euclid': Euclid, 'So3': So3}
    kerneldict = {'QuadExp': kernels.QuadExp}
    rdistdict = {'MVN': mgplvm.rdist.MVN}
    moddict = {'Sgp': models.Sgp}
    manif = params['manif'].split('(')[0]
    manif = 'So3' if manif == 'So' else manif
    m, n, d, n_z = [params[key] for key in ['m', 'n', 'd', 'n_z']]
    manif = manifdict[manif](m, d)
    kernel = kerneldict[params['kernel']](n, manif.distance)
    ref_dist = rdistdict[params['rdist']](m, d)
    mod = moddict[params['model']](kernel, manif, n, m, n_z, kernel, ref_dist)
    mod_params = torch.load(fname + '.torch')
    mod.load_state_dict(mod_params)
    mod.to(device)
    return mod, params
