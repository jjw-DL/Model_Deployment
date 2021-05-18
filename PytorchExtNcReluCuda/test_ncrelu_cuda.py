import torch
import ncrelu_cuda

a = torch.randn(4,3)
print("a:\n", a)

b = ncrelu_cuda.ncrelu_forward_cuda(a)
print("b:\n", b)

a = a.cuda()
c = ncrelu_cuda.ncrelu_forward_cuda(a)
print("c:\n", c)