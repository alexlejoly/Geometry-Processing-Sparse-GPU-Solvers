# Geometry Processing Sparse GPU Solvers

Python Library of Sparse GPU Solvers for Geometry Processing developped in the context of the COMP 401 course at McGill University (Project in Computer Science and Biology).

Developped under the supervision of Prof. Sylvain Baillet at the Montréal Neurological Institute-Hospital.

## Installation of dependencies

1. Make sure to use a python version >= 3.10. If not, install a suitable python version or create a suitable python virtual environment.
2. Perform `python -m pip install -r requirements.txt`, making sure to use the correct python executable.
3. Then, install `torch` by running the command given on [PyTorch's website](https://pytorch.org/get-started/locally/) after selecting your desired options, making sure to select a CUDA version compatible with your device and to modify `pip3` to `python -m pip`, making sure to use the correct python executable.

## Running Solvers

1. Make sure the root of the package has a `data.mat` file with the required operators.
2. Perform `python -m src.solvers.{desired solver}` from the root of the package, making sure to use the correct python executable.
3. Find the produced `result.mat` file at the root of the package, making sure to save it elsewhere if needed, since calling another solver rewrites the file.

## License

This project is licensed under the [MIT License](LICENSE).