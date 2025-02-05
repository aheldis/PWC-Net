# #!/usr/bin/env python

# import os

# from setuptools import setup, find_packages

# this_file = os.path.dirname(__file__)

# setup(
#     name="correlation_package",
#     version="0.1",
#     description="Correlation layer from FlowNetC",
#     url="https://github.com/jbarker-nvidia/pytorch-correlation",
#     author="Jon Barker",
#     author_email="jbarker@nvidia.com",
#     # Require cffi
#     install_requires=["cffi>=1.0.0"],
#     setup_requires=["cffi>=1.0.0"],
#     # Exclude the build files.
#     packages=find_packages(exclude=["build"]),
#     # Package where to put the extensions. Has to be a prefix of build.py
#     ext_package="",
#     # Extensions to compile
#     cffi_modules=[
#         os.path.join(this_file, "build.py:ffi")
#     ],
# )


#!/usr/bin/env python

import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension

this_file = os.path.dirname(__file__) + '/src'

setup(
    name="correlation_package",
    version="0.1",
    description="Correlation layer from FlowNetC",
    url="https://github.com/jbarker-nvidia/pytorch-correlation",
    author="Jon Barker",
    author_email="jbarker@nvidia.com",
    install_requires=["torch>=1.0.0"],  # Add PyTorch as a dependency
    setup_requires=["torch>=1.0.0"],
    packages=find_packages(exclude=["build"]),
    ext_modules=[
        CppExtension(
            name="",  # Name of the compiled extension
            sources=[os.path.join(this_file, "corr.c"),os.path.join(this_file, "corr1d_cuda.c"),os.path.join(this_file, "corr1d.c")],  # Add your source files here
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)