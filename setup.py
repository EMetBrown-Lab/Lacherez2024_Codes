from setuptools import setup
from Cython.Build import cythonize
import numpy

module_name = 'tm_sinusoid_trap_pack'

setup(
    name=module_name,
    ext_modules=cythonize(f"{module_name}.pyx"),
    include_dirs=[numpy.get_include()]
)


#python3 setup.py build_ext --inplace