import torch
import utils

Mesh, alpha = utils.init('build-field')

e1 = Mesh.h1 / Mesh.h1.norm(dim=1, keepdim=True)
e2 = torch.linalg.cross(Mesh.face_normal, e1)
del Mesh

field = e1 * torch.cos(alpha).T.unsqueeze(dim=2) + e2 * torch.sin(alpha).T.unsqueeze(dim=2)
del e1, e2, alpha

result = {
    'field': field.cpu().numpy()
}

utils.savemat('result.mat', result)