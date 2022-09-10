from distutils.core import setup,Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension("loadspec", ["./loadspec.pyx"],include_dirs=[np.get_include()])
setup(
    name="loadspec",
    ext_modules=cythonize(ext,compiler_directives={'language_level' : 3},annotate=True),
    )
