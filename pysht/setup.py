from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
from setuptools.command.install import install
import subprocess
import os, sys

def get_virtualenv_path():
    """Used to work out path to install compiled binaries to."""
    if hasattr(sys, 'real_prefix'):
        return sys.prefix

    if hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
        return sys.prefix

    if 'conda' in sys.prefix:
        return sys.prefix

    return None

def compile_and_install_software():
    """Used the subprocess module to compile/install the C software."""
    src_path = '.'

    # compile the software
    cmd = "./configure CFLAGS='-03 -w -fPIC'"
    venv = get_virtualenv_path()
    if venv:
        cmd += ' --prefix=' + os.path.abspath(venv)
    subprocess.check_call(cmd, cwd=src_path, shell=True)

    # install the software (into the virtualenv bin dir if present)
    # subprocess.check_call('make install', cwd=src_path, shell=True)

compile_and_install_software()

# Define the CUDA extension module
extensions = [
    Extension(
        "python_wrapper",
        sources=["python_wrapper.pyx"],  # Include your CUDA kernel file here
        language="c++",  # Specify the language as C++
        extra_objects=["kernel2.o"],  # Specify the CUDA object file
        extra_compile_args=["nvcc", "--gpu-architecture=sm_30"],  # Set CUDA compilation flags
        include_dirs=[np.get_include()],  # Include numpy headers
    )
]

# Set up the package
setup(
    name="python_wrapper",
    ext_modules=cythonize(extensions))