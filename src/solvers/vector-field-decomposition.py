from .. import utils
from scipy.io import savemat

Mesh, DEC, omega = utils.init('vector-field-decomposition')

utils.noBoundaries(Mesh)
del Mesh

alpha = utils.cholesky(DEC.A, DEC.d0.T @ DEC.hodge1 @ omega)
del DEC.A, DEC.hodge1

dAlpha = DEC.d0 @ alpha
del DEC, alpha

deltaBeta = omega - dAlpha
del omega

result = {
    'dAlpha': dAlpha.cpu().numpy(),
    'deltaBeta': deltaBeta.cpu().numpy()
}

savemat('result.mat', result)