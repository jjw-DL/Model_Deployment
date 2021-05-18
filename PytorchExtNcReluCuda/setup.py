from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ncrelu_cuda',
    ext_modules=[
        CUDAExtension('ncrelu_cuda', [
            'ncrelu_cuda.cpp', 
            'ncrelu_cuda_kernel.cu',             # 需要包含.cpp和.cu文件
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)