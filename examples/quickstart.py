"""Small CPU forward/backward example; not a performance configuration."""
import torch

from orbit_ssm import SSDMimoSSM, SSDMimoConfig


def main():
    torch.manual_seed(7)
    config = SSDMimoConfig(d_model=32, d_state=16, expand=1, d_conv=0, d_ssm_in=1, d_ssm_out=1, n_groups=16)
    model = SSDMimoSSM(config, execution={"backend": "torch", "kernel_options": {"mode": "modular"}})
    x = torch.randn(2, 8, 32, requires_grad=True)
    y = model(x)
    y.square().mean().backward()
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
    print(f"CPU forward/backward passed: {tuple(y.shape)}")


if __name__ == "__main__":
    main()
