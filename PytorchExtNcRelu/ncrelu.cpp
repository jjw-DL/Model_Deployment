#include <torch/extension.h>

torch::Tensor ncrelu_forward(torch::Tensor input) {
    auto pos = input.clamp_min(0);
    auto neg = input.clamp_max(0);
    return torch::cat({pos, neg}, 1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &ncrelu_forward, "NCReLU forward");
}