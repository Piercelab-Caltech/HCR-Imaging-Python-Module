from setuptools import Extension, setup
import sys, os

if sys.platform == 'darwin':
    args = ['-framework', 'Accelerate']
elif os.environ.get('HCR_MULTIPLEXED_IMAGING_BLAS'):
    args = ['-l' + os.environ['HCR_MULTIPLEXED_IMAGING_BLAS']]
else:
    args = ['-DARMA_DONT_USE_BLAS']

if sys.platform == 'win32':
    args += ['/std:c++17', '/O2']
else:
    args += ['-std=c++17', '-O3']

setup(
    ext_modules=[
        Extension(
            name="hcr_multiplexed_imaging.cpp", 
            extra_compile_args=args,
            include_dirs=['hcr_multiplexed_imaging/source/armadillo/include'],
            sources=["hcr_multiplexed_imaging/source/Module.cc"],
        ),
    ]
)
