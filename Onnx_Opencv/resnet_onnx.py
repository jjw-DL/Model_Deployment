import os 

import torch
import torch.onnx
from torch.autograd import Variable
from torchvision import models

def get_pytorch_onnx_model(original_model):
    onnx_model_path = "models"
    onnx_model_name = "resnet50.onnx"
    os.makedirs(onnx_model_path, exist_ok=True)
    full_model_path = os.path.join(onnx_model_path, onnx_model_name)

    generated_input = Variable(
        torch.randn(1, 3, 224, 224)
    )

    torch.onnx.export(
        original_model,
        generated_input,
        full_model_path,
        verbose=True,
        input_names=["input"],
        output_names=["output"],
        opset_version=11)
    
    return full_model_path

def main():
    origin_model = models.resnet50(pretrained=True)

    full_model_path = get_pytorch_onnx_model(origin_model)

    print("Pytorch ResNet-50 model was successfully converted:", full_model_path)


if __name__ == "__main__":
    main()