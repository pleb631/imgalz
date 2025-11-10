from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import pybind11
import sys
import traceback

ext_modules = [
    Extension(
        "imgalz.utils.cpp._hashfilter",
        ["imgalz/utils/cpp/_hashfilter.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["/O2", "/std:c++17"] if sys.platform.startswith("win") else ["-O3", "-std=c++17"],
    )
]

class build_ext(_build_ext):
    def run(self):
        try:
            super().run()
        except Exception:
            print("⚠️ C++ extension build failed. Falling back to Python version.")
            traceback.print_exc()

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception:
            print(f"⚠️ Failed to build extension {ext.name}, skipping...")
            traceback.print_exc()


setup(
    name="imgalz",              
    version="0.0.9.0",        
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
