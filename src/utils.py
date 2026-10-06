from cholespy import CholeskySolverD, MatrixType
from types import SimpleNamespace
from scipy.io import loadmat
from scipy.sparse import eye
import torch
import numpy
import sys

def buildEdgesMatrix(Mesh):
    e1 = torch.stack([Mesh.F[:,0], Mesh.F[:,1]], dim=1)
    e2 = torch.stack([Mesh.F[:,1], Mesh.F[:,2]], dim=1)
    e3 = torch.stack([Mesh.F[:,2], Mesh.F[:,0]], dim=1)

    e = torch.stack([e1, e2, e3], dim=1)

    sorted_e, _ = torch.sort(e, dim=2)
    sorted_e = sorted_e.reshape(-1, 2)

    #E = Edges matrix
    E, inverse = torch.unique(sorted_e, return_inverse=True, dim=0)

    #Face to edges matrix (Each face has 3 edges that make it up)
    F_E = inverse.reshape(-1, 3)

    return E, F_E

def buildHodge1(Mesh):
    cotan1, cotan2, cotan3 = buildCotans(Mesh)
    cotan = torch.cat([cotan1, cotan2, cotan3], dim=1)

    cotan_per_edge = torch.zeros(Mesh.E.shape[0], dtype=torch.float64, device="cuda")
    cotan_per_edge.scatter_add_(0, Mesh.F_E.flatten(), cotan.flatten())
    cotan_per_edge /= 2

    indices = torch.arange(Mesh.E.shape[0], dtype=torch.float64, device="cuda")
    return torch.sparse_coo_tensor(torch.stack([indices, indices]), cotan_per_edge)

def buildD0(Mesh):
    row = torch.arange(Mesh.E.shape[0], device="cuda").repeat_interleave(2)
    col = Mesh.E.flatten()
    data = torch.tensor([1,-1], device="cuda").repeat(Mesh.E.shape[0])

    return torch.sparse_coo_tensor(torch.stack([row, col], dim=0), data, dtype=torch.float64)

def buildFlowMatrix(Mesh, DEC):
    mean_edge_length = torch.linalg.norm(Mesh.V[Mesh.E[:,0]] - Mesh.V[Mesh.E[:,1]], dim=1).mean()
    t = torch.pow(mean_edge_length,2)

    return DEC.M + (DEC.A * t)

def buildX(Mesh, u):

    x = torch.linalg.cross(Mesh.face_normal, Mesh.h1) * u[Mesh.F[:,2]].T.unsqueeze(2)
    x += torch.linalg.cross(Mesh.face_normal, Mesh.h2) * u[Mesh.F[:,0]].T.unsqueeze(2)
    x += torch.linalg.cross(Mesh.face_normal, Mesh.h3) * u[Mesh.F[:,1]].T.unsqueeze(2)
    x /= Mesh.cross_norm
    x /= x.norm(dim=-1, keepdim=True)

    return x

def divX(Mesh, x):
    cotan1, cotan2, cotan3 = buildCotans(Mesh)

    contri1 = cotan1 * dot(Mesh.h1, x) + cotan3 * dot(-Mesh.h3, x)
    contri2 = cotan2 * dot(Mesh.h2, x) + cotan1 * dot(-Mesh.h1, x)
    contri3 = cotan3 * dot(Mesh.h3, x) + cotan2 * dot(-Mesh.h2, x)
    contributions = torch.cat([contri1, contri2, contri3], dim=-1)

    div = torch.zeros((Mesh.V.shape[0], x.shape[0]), dtype=torch.float64, device="cuda")
    div.scatter_add_(0, Mesh.F.reshape(-1,1).expand(-1, x.shape[0]), contributions.permute(1, 2, 0).reshape(-1, x.shape[0]))
    div /= -2
    
    return div

def buildHalfEdgeVectors(Mesh):
    h1 = Mesh.V[Mesh.F[:,1]] - Mesh.V[Mesh.F[:,0]]
    h2 = Mesh.V[Mesh.F[:,2]] - Mesh.V[Mesh.F[:,1]]
    h3 = Mesh.V[Mesh.F[:,0]] - Mesh.V[Mesh.F[:,2]]

    return h1, h2, h3

def buildCotans(Mesh):
    cotan1 = dot(Mesh.h3, -Mesh.h2) / Mesh.cross_norm
    cotan2 = dot(Mesh.h1, -Mesh.h3) / Mesh.cross_norm
    cotan3 = dot(Mesh.h2, -Mesh.h1) / Mesh.cross_norm

    return cotan1, cotan2, cotan3

def angleSum(Mesh):
    angle1 = angle(Mesh.h3, Mesh.h1)
    angle2 = angle(Mesh.h1, Mesh.h2)
    angle3 = angle(Mesh.h2, Mesh.h3)
    angles = torch.cat([angle1, angle2, angle3], dim=1)

    angleSum = torch.zeros(Mesh.V.shape[0], dtype=torch.float64, device="cuda")
    angleSum.scatter_add_(0, Mesh.F.flatten(), angles.flatten())
    angleSum = angleSum.reshape(-1,1)

    return angleSum

def cholesky(pdM, rhs):
    x = torch.zeros_like(rhs)

    #Cholespy requires coalesced matrices
    coalesced_pdM = pdM.coalesce()

    llt = CholeskySolverD(pdM.shape[0], coalesced_pdM.indices()[0], coalesced_pdM.indices()[1], coalesced_pdM.values(), MatrixType.COO)

    #cholespy's solve method only works with batches containing a max of 128 instances, this loop simply takes batches of 128 at a time from a rhs with batch size > 128 and constructs one resulting matrix of the correct batch size  
    for i in range(0, rhs.shape[1], 128):
        temp_x = x[:,i:i+128].contiguous()
        llt.solve(rhs[:,i:i+128].contiguous(), temp_x)
        x[:,i:i+128] = temp_x

    return x

def dot(u, v):
    return (u * v).sum(-1, keepdim=True)

def angle(u, v):
    return (dot(-u, v) / (u.norm(dim=1, keepdim=True) * v.norm(dim=1, keepdim=True))).clamp(-1,1).acos()

def noBoundaries(Mesh):
    if Mesh.V.shape[0] - Mesh.E.shape[0] + Mesh.F.shape[0] != 2:
        print('Mesh has boundaries', file=sys.stderr)
        sys.exit(1)

def satisfyGaussBonnet(Mesh, singularity):
    if not torch.abs(Mesh.V.shape[0] - Mesh.E.shape[0] + Mesh.F.shape[0] - singularity.sum(dim=0)).all() < 1e-8:
        print('Singularities do not add up to the euler characteristic of the mesh', file=sys.stderr)
        sys.exit(2)

#Packs all the required DEC and Mesh operators, and inputs, and each solver calls this function at the top of the file in order to separate this from the logic of the solvers' algorithms itself
def init(solver):

    #data.mat needs to be at the root
    data = loadmat('./data.mat')
    Mesh = SimpleNamespace()
    DEC = SimpleNamespace()

    if solver == 'poisson':
        A_scipy = data['A']
        A_scipy += 1e-8 * eye(A_scipy.shape[0], format=A_scipy.format)
        A_scipy = A_scipy.tocoo()
        DEC.A = torch.sparse_coo_tensor(numpy.array([A_scipy.row, A_scipy.col]), A_scipy.data, dtype=torch.float64, device="cuda")

        M_scipy = data['M']
        M_scipy = M_scipy.tocoo()
        DEC.M = torch.sparse_coo_tensor(numpy.array([M_scipy.row, M_scipy.col]), M_scipy.data, dtype=torch.float64, device="cuda") / 9216

        rho_scipy = data['rho']
        rho = torch.from_numpy(rho_scipy).to(torch.float64).to("cuda")

        return DEC, rho
    
    if solver == 'geodesic-distance':
        A_scipy = data['A']
        A_scipy += 1e-8 * eye(A_scipy.shape[0], format=A_scipy.format)
        A_scipy = A_scipy.tocoo()
        DEC.A = torch.sparse_coo_tensor(numpy.array([A_scipy.row, A_scipy.col]), A_scipy.data, dtype=torch.float64, device="cuda")

        M_scipy = data['M']
        M_scipy = M_scipy.tocoo()
        DEC.M = torch.sparse_coo_tensor(numpy.array([M_scipy.row, M_scipy.col]), M_scipy.data, dtype=torch.float64, device="cuda") / 9216

        V_scipy = data['V']
        Mesh.V = torch.from_numpy(V_scipy).to(torch.float64).to("cuda") / 96

        F_scipy = data['F']
        Mesh.F = torch.from_numpy(F_scipy).to(torch.int32).to("cuda") - 1

        Mesh.E, _ = buildEdgesMatrix(Mesh)

        Mesh.h1, Mesh.h2, Mesh.h3 = buildHalfEdgeVectors(Mesh)
        cross = torch.linalg.cross(-Mesh.h3, Mesh.h1)
        Mesh.cross_norm = cross.norm(dim=1, keepdim=True)
        Mesh.face_normal = cross / Mesh.cross_norm

        delta_scipy = data['delta']
        delta = torch.from_numpy(delta_scipy).to(torch.float64).to("cuda")

        return Mesh, DEC, delta
    
    if solver == 'vector-field-decomposition':
        A_scipy = data['A']
        A_scipy += 1e-8 * eye(A_scipy.shape[0], format=A_scipy.format)
        A_scipy = A_scipy.tocoo()
        DEC.A = torch.sparse_coo_tensor(numpy.array([A_scipy.row, A_scipy.col]), A_scipy.data, dtype=torch.float64, device="cuda")

        V_scipy = data['V']
        Mesh.V = torch.from_numpy(V_scipy).to(torch.float64).to("cuda") / 96

        F_scipy = data['F']
        Mesh.F = torch.from_numpy(F_scipy).to(torch.int32).to("cuda") - 1

        Mesh.h1, Mesh.h2, Mesh.h3 = buildHalfEdgeVectors(Mesh)
        cross = torch.linalg.cross(-Mesh.h3, Mesh.h1)
        Mesh.cross_norm = cross.norm(dim=1, keepdim=True)

        Mesh.E, Mesh.F_E = buildEdgesMatrix(Mesh)
        DEC.d0 = buildD0(Mesh)
        DEC.hodge1 = buildHodge1(Mesh)

        omega_scipy = data['omega']
        omega = torch.from_numpy(omega_scipy).to(torch.float64).to("cuda")

        return Mesh, DEC, omega
    
    if solver == 'trivial-connections':
        A_scipy = data['A']
        A_scipy += 1e-8 * eye(A_scipy.shape[0], format=A_scipy.format)
        A_scipy = A_scipy.tocoo()
        DEC.A = torch.sparse_coo_tensor(numpy.array([A_scipy.row, A_scipy.col]), A_scipy.data, dtype=torch.float64, device="cuda")

        V_scipy = data['V']
        Mesh.V = torch.from_numpy(V_scipy).to(torch.float64).to("cuda") / 96

        F_scipy = data['F']
        Mesh.F = torch.from_numpy(F_scipy).to(torch.int32).to("cuda") - 1

        Mesh.h1, Mesh.h2, Mesh.h3 = buildHalfEdgeVectors(Mesh)
        cross = torch.linalg.cross(-Mesh.h3, Mesh.h1)
        Mesh.cross_norm = cross.norm(dim=1, keepdim=True)

        Mesh.E, Mesh.F_E = buildEdgesMatrix(Mesh)
        DEC.d0 = buildD0(Mesh)
        DEC.hodge1 = buildHodge1(Mesh)

        singularity_scipy = data['singularity']
        singularity = torch.from_numpy(singularity_scipy).to(torch.float64).to("cuda")

        return Mesh, DEC, singularity
    
    if solver == 'build-field':
        V_scipy = data['V']
        Mesh.V = torch.from_numpy(V_scipy).to(torch.float64).to("cuda") / 96

        F_scipy = data['F']
        Mesh.F = torch.from_numpy(F_scipy).to(torch.int32).to("cuda") - 1

        Mesh.h1, Mesh.h2, Mesh.h3 = buildHalfEdgeVectors(Mesh)
        cross = torch.linalg.cross(-Mesh.h3, Mesh.h1)
        cross_norm = cross.norm(dim=1, keepdim=True)
        Mesh.face_normal = cross / cross_norm

        #due to the difficulty of accelerating parts of the direction field design algorithm, what comes after trivial connections 
        #simply needs to be performed using geometry-processing-js's logic on CPU, and the resulting alpha can be fed to this build-field GPU solver, since build-field can be acelerated
        alpha_scipy = data['alpha']
        alpha = torch.from_numpy(alpha_scipy).to(torch.float64).to("cuda")

        return Mesh, alpha
