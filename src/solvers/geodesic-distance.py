import torch
from .. import utils
from scipy.io import savemat

Mesh, DEC, delta = utils.init('geodesic-distance')

flow = utils.buildFlowMatrix(Mesh, DEC)

u = utils.cholesky(flow, delta)
del flow, delta

x = utils.buildX(Mesh, u)
del u

div = utils.divX(Mesh, x)
del Mesh, x, DEC.M

phi = utils.cholesky(DEC.A, div)
del DEC, div

phi -= torch.min(phi, 0, True).values

result = {
    'phi': phi.cpu().numpy()
}

savemat('result.mat', result)