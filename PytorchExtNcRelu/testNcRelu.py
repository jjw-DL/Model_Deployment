import torch
import ncrelu_cpp

a = torch.randn(4,3)
print("a:\n", a)

b = ncrelu_cpp.forward(a)
print("b:\n", b)

a = a.cuda()
c = ncrelu_cpp.forward(a)
print("c:\n", c)
