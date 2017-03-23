from distutils.core import setup
from Cython.Build import cythonize

setup(
      name='boxlib',
      ext_modules=cythonize("boxlib.pyx"),
)