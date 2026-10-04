# Geometry Processing Sparse GPU Solvers
Python Library of Sparse GPU Solvers for Geometry Processing developed in the context of the [COMP 401 course](https://coursecatalogue.mcgill.ca/archive/2025-2026/courses/comp-401/index.html) at McGill University, Montréal, QC (Project in Computer Science and Biology).

Developed under the supervision of Prof. Sylvain Baillet at the Montréal Neurological Institute-Hospital.

## About
Multiple mesh geometry processing algorithms can be extremely useful for the study of dynamic brain activity. However, in our case, the ones that were targeted for acceleration for the needs of Brainstorm were the scalar Poisson equation, the heat method for geodesic distance, Hodge decomposition for vector fields, and trivial connections for direction field design, all described and implemented for CPU in the geometry-processing-js repository. It is described as a “fast and flexible framework for 3D geometry processing [...]”, and was created by the Geometry Collective research group out of Carnegie Mellon University [1].

The scalar Poisson equation Kϕ = M (p̅ − ρ) aims to solve ϕ, a matrix representing a gravitational or electric potential at each vertex, from ρ, a matrix of the same shape representing charge density. K is the stiffness matrix and M is the mass matrix and together, they encode the Laplacian operator, which plays an important role in many geometry equations [2, 1].

The heat method is a multi-step algorithm that efficiently calculates geodesic distances from one point to all other points on a surface mesh. The first step is to find the heat allowed to diffuse on the mesh from a point by solving for u in F u = δ, where δ is a matrix containing a 1 at the source vertex and 0s elsewhere, and where F is the flow matrix. The resulting u is then used to build a vector field X, where each face has a vector, by solving X = −∇u/∥∇u∥. The integrated divergence is then computed on X to get ∇X, which is used as the right side of the scalar Poisson equation Kϕ = ∇X. Finally, every value inside ϕ is shifted by the minimum value situated at the source to get the geodesic distances [1].

Hodge decomposition is useful in order to decompose a vector field on edges into its exact, co-exact and harmonic components. In our case, the harmonic component is absent since it always equals zero on sphere meshes [9], which is the case of brain meshes [8]. The exact component dα is found by solving for α in Kα = dT ⋆1 ω, where d is a DEC operator encoding exterior derivatives on 0-forms, where ⋆1 is the Hodge star DEC operator on 1-forms, and where ω is the input vector field, before multiplying d to α. More trivially, the co-exact component δβ is then simply found by doing δβ = ω − dα [1].

Finally, trivial connections is an algorithm for constructing connections on discrete surfaces, in our case a mesh, that are smooth everywhere except for near an indicated source point and end point. This algorithm is useful, since it can be used to design rotationally symmetric direction fields [10]. The algorithm computes connections by solving a linear system built by using angle defects, d and ⋆1 [1].

## Installation of dependencies
1. Make sure to use a python version >= 3.10.
2. Perform `python -m pip install -r requirements.txt` from the root of the package, making sure to use the correct python executable.
3. Then, install `torch` by running the command given on [PyTorch's website](https://pytorch.org/get-started/locally/) after selecting your desired options, making sure to select a CUDA version compatible with your device, to replace `pip3` with `python -m pip`, and to use the correct python executable.

## Running Solvers
1. Make sure the root of the package has a `data.mat` file with the required operators.
2. Perform `python -m src.solvers.{desired solver}` from the root of the package, making sure to use the correct python executable.
   e.g. `python -m src.solvers.poisson` if wanting to execute the poisson solver script.
3. Find the produced `result.mat` file at the root of the package, making sure to save it elsewhere if needed, since calling another solver rewrites the file.

## References
[1] Geometry Collective. “geometry-processing-js.” GitHub. Accessed: Apr. 12, 2026. [Online]. Available: https://github.com/geometrycollective/geometry-processing-js

[2] K. Crane, “The Laplacian,” in DISCRETE DIFFERENTIAL GEOMETRY: AN APPLIED INTRODUCTION, 2025, pp. 101-116. [Online]. Available: https://www.cs.cmu.edu/~kmcrane/Projects/DDG/paper.pdf

## License
This project is licensed under the [MIT License](LICENSE).
