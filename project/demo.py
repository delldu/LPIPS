import image_likeness

from PIL import Image
import torch
import torchvision.transforms as T
import time


def test_model(model_name):
    device = "cuda"
    model = image_likeness.get_model(model_name)
    model = model.to(device)
    model.eval()

    ex_ref = T.ToTensor()(Image.open("output/ex_ref.png"))
    ex_p0 = T.ToTensor()(Image.open("output/ex_p0.png"))
    ex_p1 = T.ToTensor()(Image.open("output/ex_p1.png"))

    ex_ref = ex_ref.unsqueeze(0).to(device)
    ex_p0 = ex_p0.unsqueeze(0).to(device)
    ex_p1 = ex_p1.unsqueeze(0).to(device)

    # model accept input as [0, 1]
    start_time = time.time()
    for i in range(100):
        with torch.no_grad():
            ex_d0 = model(ex_ref, ex_p0)
            ex_d1 = model(ex_ref, ex_p1)
            ex_d2 = model(ex_ref, ex_ref)
            ex_d3 = model(ex_p0, ex_p1)
            ex_d4 = model(ex_p1, ex_p0)

    print(f"{model_name} spend time: { time.time() - start_time} seconds for 10 times",)
    print("    Loss: ref-p0: %.3f > ref-p1: %.3f" % (ex_d0, ex_d1))
    print("    Loss: ref-ref: %.3f == 0.00" % (ex_d2))
    print("    Loss: p0-p1: %.3f == p1-p0: %.3f" % (ex_d3, ex_d4))


if __name__ == "__main__":
    """Test."""

    test_model("alex")
    test_model("squeeze")
    test_model("vgg16")
