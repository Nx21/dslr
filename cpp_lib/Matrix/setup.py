from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import sys

class BuildExt(build_ext):
    def build_extensions(self):
        import pybind11
        for ext in self.extensions:
            ext.include_dirs.append(pybind11.get_include())
        super().build_extensions()

ext = Extension(
    "matrix",
    sources=[
        "src/python_bindings.cpp",
        "src/constructors.cpp",
        "src/functions.cpp",
        "src/operations.cpp",
        "src/property.cpp",
        "src/vectorOperations.cpp",
    ],
    include_dirs=["include"],
    language="c++",
    extra_compile_args=["-std=c++17", "-O2"],
)

setup(
    name="matrix",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExt},
    setup_requires=["pybind11>=2.6.0"],
)