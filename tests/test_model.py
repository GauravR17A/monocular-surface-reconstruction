import torch

from msr.models.height_net import HeightNet


def test_height_model_preserves_requested_output_size():
    model = HeightNet(
        backbone="resnet18",
        pretrained=False,
        decoder_channels=(128, 64, 32, 16, 8),
        auxiliary_building_head=True,
    )
    image = torch.randn(1, 3, 65, 71)

    with torch.inference_mode():
        output = model(image)

    assert output["height"].shape == (1, 1, 65, 71)
    assert output["building_logits"].shape == (1, 1, 65, 71)
    assert torch.all(output["height"] >= 0)

