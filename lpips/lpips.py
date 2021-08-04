import os
import inspect

import torch
import torch.nn as nn
import numpy as np
import model_helper as pn

import onnx
import onnxruntime

from PIL import Image
import torchvision.transforms as transforms

import pdb


def upsample(in_tens, out_HW=(64, 64)):  # assumes scale factor is same for H and W
    in_H, in_W = in_tens.shape[2], in_tens.shape[3]
    return nn.Upsample(size=out_HW, mode="bilinear", align_corners=False)(in_tens)


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)


# Learned perceptual metric
class LPIPS(nn.Module):
    def __init__(self, net="alex"):
        super(LPIPS, self).__init__()

        version = "0.1"

        self.pnet_type = net
        self.version = version
        self.scaling_layer = ScalingLayer()

        if self.pnet_type in ["vgg", "vgg16"]:
            net_type = pn.vgg16
            self.channels = [64, 128, 256, 512, 512]
        elif self.pnet_type == "alex":
            net_type = pn.alexnet
            self.channels = [64, 192, 384, 256, 256]
        elif self.pnet_type == "squeeze":
            net_type = pn.squeezenet
            self.channels = [64, 128, 256, 384, 384, 512, 512]
        self.L = len(self.channels)

        self.net = net_type(pretrained=True, requires_grad=False)

        self.lin0 = NetLinLayer(self.channels[0])
        self.lin1 = NetLinLayer(self.channels[1])
        self.lin2 = NetLinLayer(self.channels[2])
        self.lin3 = NetLinLayer(self.channels[3])
        self.lin4 = NetLinLayer(self.channels[4])
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        if self.pnet_type == "squeeze":  # 7 layers for squeezenet
            self.lin5 = NetLinLayer(self.channels[5])
            self.lin6 = NetLinLayer(self.channels[6])
            self.lins += [self.lin5, self.lin6]
        self.lins = nn.ModuleList(self.lins)

        model_path = os.path.abspath(
            os.path.join(
                inspect.getfile(self.__init__),
                "..",
                "weights/v%s/%s.pth" % (version, net),
            )
        )

        print("Loading model from: %s" % model_path)
        self.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)

    def forward(self, input):
        image_0 = input[0:1]
        image_1 = input[1:2]

        # Move input from [0,1] [-1, +1]
        image_0 = 2 * image_0 - 1
        image_1 = 2 * image_1 - 1

        input_0 = self.scaling_layer(image_0)
        input_1 = self.scaling_layer(image_1)

        output_0 = self.net.forward(input_0)
        output_1 = self.net.forward(input_1)
        features_0, features_1, difference = {}, {}, {}

        for k in range(self.L):
            features_0[k] = normalize_tensor(output_0[k])
            features_1[k] = normalize_tensor(output_1[k])

            difference[k] = (features_0[k] - features_1[k]) ** 2

        res = [
            upsample(self.lins[k](difference[k]), out_HW=image_0.shape[2:])
            for k in range(self.L)
        ]

        val = res[0]
        for l in range(1, self.L):
            val += res[l]

        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer(
            "shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None]
        )
        self.register_buffer(
            "scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None]
        )

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in, chn_out=1):
        super(NetLinLayer, self).__init__()

        layers = [
            nn.Dropout(),
        ]
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def model_setenv():
    """Setup environ  ..."""

    import random
    import time

    random_seed = int(time.time() % 1000)
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Set default device to avoid exceptions
    if os.environ.get("DEVICE") != "cuda" and os.environ.get("DEVICE") != "cpu":
        os.environ["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"

    if os.environ["DEVICE"] == "cuda":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    print("Running Environment:")
    print("----------------------------------------------")
    print("  PWD: ", os.environ["PWD"])
    print("  DEVICE: ", os.environ["DEVICE"])


def model_device():
    """Please call after model_setenv."""
    return torch.device(os.environ["DEVICE"])


def get_model():
    """Create model."""
    model_setenv()
    model = LPIPS(net="vgg")
    return model


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def onnx_model_load(onnx_file):
    sess_options = onnxruntime.SessionOptions()
    # sess_options.log_severity_level = 0

    # Set graph optimization level
    sess_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    )

    onnx_model = onnxruntime.InferenceSession(onnx_file, sess_options)
    # onnx_model.set_providers(['CUDAExecutionProvider'])
    print(
        "Onnx model engine: ",
        onnx_model.get_providers(),
        "Device: ",
        onnxruntime.get_device(),
    )

    return onnx_model


def onnx_model_forward(onnx_model, input):
    onnxruntime_inputs = {onnx_model.get_inputs()[0].name: to_numpy(input)}
    onnxruntime_outputs = onnx_model.run(None, onnxruntime_inputs)
    return torch.from_numpy(onnxruntime_outputs[0])


def export_onnx():
    """Export onnx model."""

    onnx_file_name = "output/image_lploss.onnx"
    dummy_input = torch.randn(2, 3, 64, 64).cuda()

    # 1. Create and load model.
    torch_model = get_model()
    torch_model = torch_model.cuda()
    torch_model.eval()

    # 2. Model export
    print("Export decoder model ...")

    input_names = ["input"]
    output_names = ["output"]
    torch.onnx.export(
        torch_model,
        dummy_input,
        onnx_file_name,
        input_names=input_names,
        output_names=output_names,
        verbose=True,
        opset_version=11,
        keep_initializers_as_inputs=False,
        export_params=True,
    )

    # 3. Optimize model
    print("Checking model ...")
    onnx_model = onnx.load(onnx_file_name)
    onnx.checker.check_model(onnx_model)
    onnx.helper.printable_graph(onnx_model.graph)
    # https://github.com/onnx/optimizer

    # 4. Visual model
    # python -c "import netron; netron.start('output/image_lploss.onnx')"


def verify_onnx():
    """Verify onnx model."""

    onnx_file_name = "output/image_lploss.onnx"
    torch_model = get_model()
    torch_model.eval()
    onnxruntime_engine = onnx_model_load(onnx_file_name)

    dummy_input = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        torch_output = torch_model(dummy_input)

    onnxruntime_inputs = {
        onnxruntime_engine.get_inputs()[0].name: to_numpy(dummy_input),
    }

    onnxruntime_outputs = onnxruntime_engine.run(None, onnxruntime_inputs)
    np.testing.assert_allclose(
        to_numpy(torch_output), onnxruntime_outputs[0], rtol=1e-02, atol=1e-02
    )
    print("Onnx model tested with ONNXRuntime, result sounds good !")


def test_sample():
    """Test."""
    model_setenv()
    model = get_model()
    device = model_device()
    model = model.to(device)
    model.eval()

    totensor = transforms.ToTensor()

    ex_ref = totensor(Image.open("output/ex_ref.png"))
    ex_p0 = totensor(Image.open("output/ex_p0.png"))
    ex_p1 = totensor(Image.open("output/ex_p1.png"))
    ex_ref = ex_ref.unsqueeze(0).to(device)
    ex_p0 = ex_p0.unsqueeze(0).to(device)
    ex_p1 = ex_p1.unsqueeze(0).to(device)

    # model default accept input as [0, 1]
    with torch.no_grad():
        ex_d0 = model(torch.cat([ex_ref, ex_p0], dim=0))
        ex_d1 = model(torch.cat([ex_ref, ex_p1], dim=0))
        ex_d2 = model(torch.cat([ex_ref, ex_ref], dim=0))
        ex_d3 = model(torch.cat([ex_p0, ex_p1], dim=0))
        ex_d4 = model(torch.cat([ex_p1, ex_p0], dim=0))

    print(
        "Loss: (%.3f, %.3f, %.3f, %.3f, %.3f)"
        % (ex_d0.mean(), ex_d1.mean(), ex_d2.mean(), ex_d3.mean(), ex_d4.mean())
    )


if __name__ == "__main__":
    """Test Tools ..."""
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--export", help="Export onnx model", action="store_true")
    parser.add_argument("--verify", help="Verify onnx model", action="store_true")
    parser.add_argument("--test", help="Test", action="store_true")

    parser.add_argument("--output", type=str, default="output", help="output folder")

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.export:
        export_onnx()

    if args.verify:
        verify_onnx()

    if args.test:
        test_sample()
