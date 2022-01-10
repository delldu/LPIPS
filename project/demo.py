import image_likeness

from PIL import Image
import torch
import torchvision.transforms as T
import time

if __name__ == "__main__":
    """Test."""

    device = "cuda"
    model = image_likeness.get_model()
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
            ex_d0 = model(torch.cat([ex_ref, ex_p0], dim=0))
            ex_d1 = model(torch.cat([ex_ref, ex_p1], dim=0))
            ex_d2 = model(torch.cat([ex_ref, ex_ref], dim=0))
            ex_d3 = model(torch.cat([ex_p0, ex_p1], dim=0))
            ex_d4 = model(torch.cat([ex_p1, ex_p0], dim=0))

    print("Spend time: ", time.time() - start_time)
    print(
        "Loss: (ref-p0: %.3f > ref-p1: %.3f)" % (ex_d0.mean(), ex_d1.mean())
    )

    print(
        "Loss: (ref-ref: %.3f) == 0.00" % (ex_d2.mean())
    )

    print(
        "Loss: (p0-p1: %.3f == p1-p0: %.3f)" % (ex_d3.mean(), ex_d4.mean())
    )

