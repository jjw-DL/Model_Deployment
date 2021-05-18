## 编写一个 C++ 拓展

### 编写setup.py

利用python中提供的setuptools模块完成事先编译流程，将写有算子的C++文件，编译成为一个动态链接库（在Linux平台是一个.so后缀文件），可以让python调用其中实现的函数功能。

```python
from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='ncrelu_cpp',  # 编译后的链接库名称
    ext_modules=[cpp_extension.CppExtension(
        'ncrelu_cpp',['ncrelu.cpp']  # 待编译文件，及编译函数
    )],
    cmdclass={  # 执行编译命令设置
        'build_ext':cpp_extension.BuildExtension
    }
)
```

`CppExtension`是`setuptools.Extension`的一个便利的包装器(wrapper），它传递正确的包含路径并将扩展语言设置为 C++。 等效的普通`setuptools`代码像下面这样简单：

```python
setuptools.Extension(
   name='ncrelu_cpp',
   sources=['ncrelu_cpp.cpp'],
   include_dirs=torch.utils.cpp_extension.include_paths(),
   language='c++')
```

`BuildExtension`执行许多必需的配置步骤和检查，并在混合 C++/CUDA 扩展的情况下管理混合编译。

### 编写 C++ 文件（ncrelu_cpp.cpp）

```c++
#include <torch/extension.h>					// 头文件引用部分 

torch::Tensor ncrelu_forward(torch::Tensor input) {
    auto pos = input.clamp_min(0);				       // 具体实现部分
    auto neg = input.clamp_max(0);
    return torch::cat({pos, neg}, 1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {	// 绑定部分
    m.def("forward", &ncrelu_forward, "NCReLU forward");
}
```

以上代码包含了三个部分:

- **头文件引用部分：**这里包含了`torch/extension.h`头文件，是编写Pytorch的C++扩展时必须包含的一个文件。它基本上囊括了实现中所需要的所有依赖，包含了**ATen库**，**pybind11**和二者之间的交互。其中ATen是Pytorch底层张量运算库，负责实现具体张量操作运算；pybind11是实现C++代码到python的**绑定**（binding），可以在python里调用C++函数。

- **具体实现部分：**函数返回类型和传递参数类型均是`torch::Tensor`类，这种对象不仅包含了数据，还附属了诸多运算操作。因此我们可以看到在下面实现方式类似于python中使用Pytorch张量运算操作一样，**可以直接调用如截取操作clamp和拼接操作cat等**，非常简洁已读且方便。
- **绑定部分：**只需要在`m.def`中传入参数，分别是绑定到python的函数名称，需绑定的C++函数引用，以及一个简短的函数说明字符串，用来添加到python函数中的`__doc__`成员名称中。
- 宏`TORCH_EXTENSION_NAME`。torch 的扩展部分将会把它定义为你在`setup.py`脚本中为扩展名命名的名称。在这种情况下，`TORCH_EXTENSION_NAME`的值将为“ncrelu_cpp”。这是为了避免必须在两个地方(构建脚本和 C++ 代码中）维护扩展名，

将以上两个文件放在同一文件夹下，然后进行编译。使用`python setup.py build_ext --inplace`命令，如果一切正常将会在文件夹下产生类似`ncrelu_cpp.cpython-37m-x86_64-linux-gnu.so`动态链接文件

### python调用c++

首先运行 `import torch` ，因为这将解析动态链接器必须看到的一些符号

```python
>> import torch
>> import ncrelu_cpp
>> a = torch.randn(4, 3)
>> a
tensor([[ 0.5921,  0.3207,  0.7690],
        [ 1.4514, -0.8942,  0.9039],
        [-0.3262, -0.1610,  0.6137],
        [ 0.7824, -1.8527,  0.2844]])
>> b = ncrelu_cpp.forward(a)
>> b
tensor([[ 0.5921,  0.3207,  0.7690,  0.0000,  0.0000,  0.0000],
        [ 1.4514,  0.0000,  0.9039,  0.0000, -0.8942,  0.0000],
        [ 0.0000,  0.0000,  0.6137, -0.3262, -0.1610,  0.0000],
        [ 0.7824,  0.0000,  0.2844,  0.0000, -1.8527,  0.0000]])
```

这里从结果可以看出，所实现的ncrelu.cpp中的forward函数正确的将输入中大于0和小于0两部分拼接在了一起。

而且由于我们是利用Pytorch中ATen张量库封装的高层操作，**是一种与运行设备无关的代码抽象**，因此上面所实现的函数可以**直接应用于GPU上进行计算**，只需要将输入迁移至GPU上即可。

```c++
>> a = a.cuda()
>> c = ncrelu_cpp.forward(a)
>> c
tensor([[ 0.5921,  0.3207,  0.7690,  0.0000,  0.0000,  0.0000],
        [ 1.4514,  0.0000,  0.9039,  0.0000, -0.8942,  0.0000],
        [ 0.0000,  0.0000,  0.6137, -0.3262, -0.1610,  0.0000],
        [ 0.7824,  0.0000,  0.2844,  0.0000, -1.8527,  0.0000]],
       device='cuda:0')
```

### JIT 编译扩展

JIT 编译机制通过在 PyTorch 的 API 中调用一个名为`torch.utils.cpp_extension.load()`的简单函数，为你提供了一种编译和加载扩展的方法。

```python
from torch.utils.cpp_extension import load

lltm = load(name="ncrelu_cpp", sources=["ncrelu_cpp.cpp"])
```

在这里，我们为函数提供与`setuptools`相同的信息。 在后台，将执行以下操作：

1. 创建临时目录 `/tmp/torch_extensions/ncrelu_cpp`
2. 将一个 [Ninja](https://ninja-build.org/) 构建文件发送到该临时目录，
3. 将源文件编译为共享库
4. 将此共享库导入为 Python 模块

实际上，如果你将`verbose = True`参数传递给`cpp_extension.load(）`，该过程在进行的过程中将会告知你：

```c++
Using /tmp/torch_extensions as PyTorch extensions root...
Creating extension directory /tmp/torch_extensions/ncrelu_cpp...
Emitting ninja build file /tmp/torch_extensions/lltm/build.ninja...
Building extension module ncrelu_cpp...
Loading extension module ncrelu_cpp...
```

生成的 Python 模块与 setuptools 生成的完全相同，但不需要维护单独的`setup.py`构建文件。如果你的设置更复杂并且你确实需要`setuptools`的全部功能，那么你可以编写自己的`setup.py`——但在很多情况下，这种JIT的方式就已经完全够用了。第一次运行此行代码时，将耗费一些时间，因为扩展正在后台进行编译。由于我们使用 Ninja 构建系统来构建源代码，因此重新编译的工作量是不断增加的，而当你第二次运行 Python 模块进行重新加载扩展时速度就会快得多了，而且如果你没有对扩展的源文件进行更改，需要的开销也将会很低。
