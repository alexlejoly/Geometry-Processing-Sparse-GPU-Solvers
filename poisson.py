import torch
import utils

DEC, rho = utils.init('poisson')

totalRho = (DEC.M @ rho).sum(dim=0, keepdim=True)
totalArea = DEC.M.sum()
rhoBar = torch.ones((DEC.M.shape[0],1), dtype=torch.float64) * (totalRho / totalArea)
del totalRho, totalArea

rhs = DEC.M @ (rhoBar - rho)
del DEC.M, rhoBar, rho

phi = utils.cholesky(DEC.A, rhs)
del DEC, rhs

result = {
    'phi': phi.cpu().numpy()
}

utils.savemat('result.mat', result)