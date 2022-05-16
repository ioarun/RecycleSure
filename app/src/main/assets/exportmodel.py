import torchvision

model_quantized = torchvision.models.quantization.mobilenet_v2(pretrained=True, quantize=True)
import torch

dummy_input = torch.rand(1, 3, 224, 224)
torchscript_model = torch.jit.trace(model_quantized, dummy_input)
from torch.utils.mobile_optimizer import optimize_for_mobile
torchscript_model_optimized = optimize_for_mobile(torchscript_model)
torch.jit.save(torchscript_model_optimized, "mobilenetv2_quantized.pt")
