from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension('cytmodel.cytmodel', ['cytmodel/cytmodel.pyx'], include_dirs = [np.get_include()]),
]
setup(
    name="cytmodel",
    version="0.1",
    maintainer= "Bhargav Teja Nallapu",
    maintainer_email="bhargav.teja@research.iiit.ac.in",
    install_requires=['numpy', 'cython'],
    license = "BSD License",
    packages=['cytmodel'],
    ext_modules = cythonize(extensions)
)

