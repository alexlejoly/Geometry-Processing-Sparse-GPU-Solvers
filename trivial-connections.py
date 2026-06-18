import torch
import utils

Mesh, DEC, singularity = utils.init('trivial-connections')

utils.noBoundaries(Mesh)
utils.satisfyGaussBonnet(Mesh, singularity)

angleSum = utils.angleSum(Mesh)
del Mesh

angleDefect = 2 * torch.pi - angleSum
del angleSum

rhs = -angleDefect + 2 * torch.pi * singularity
del angleDefect, singularity

betaTilde = utils.cholesky(DEC.A, rhs)
del DEC.A, rhs

phi = DEC.hodge1 @ DEC.d0 @ betaTilde
del DEC, betaTilde

result = {
    'phi': phi.cpu().numpy()
}

utils.savemat('result.mat', result)