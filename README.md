# Geometry Processing Sparse GPU Solvers
Python Library of Sparse GPU-Accelerated Solvers for Geometry Processing developed in the context of the [COMP 401 course](https://coursecatalogue.mcgill.ca/archive/2025-2026/courses/comp-401/index.html) at McGill University, Montréal, QC (Project in Computer Science and Biology).

Developed under the supervision of Prof. Sylvain Baillet at the Montréal Neurological Institute-Hospital.

Based on the `poisson-problem`, `geodesic-distance`, `vector-field-decomposition` and `direction-field-design` projects developed by GeometryCollective in [geometry-processing-js](https://github.com/GeometryCollective/geometry-processing-js/tree/master).

## Project structure

```
Geometry-Processing-Sparse-GPU-Solvers/
├── src/
│   ├── utils.py
│   └── solvers/
│       ├── poisson.py                                  
│       ├── geodesic-distance.py       
│       ├── vector-field-decomposition.py
│       ├── trivial-connections.py
│       └── build-field.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

### Available solvers

-`poisson.py` solves a scalar poisson problem on a surface mesh.
-`geodesic-distance.py` computes the geodesic distance using the heat method on a surface mesh.
-`vector-field-decomposition.py` computes the hodge decomposition of a vector field on a surface mesh.
-`trivial-connections.py` computes a smooth 1-form vector field using the trivial connections algorithm on a surface mesh (direction field design).
-`build-field.py` builds the direction field using the resulting trivial connections on a surface mesh (direction field design).

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
