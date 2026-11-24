import pytest
import torch

from efficient_kan.kan import KANLinear as FixedLayer
from efficient_kan.konts_kan import KANLinear, Knots_KAN
from efficient_kan.spline import basis, divide_or_zero, second_derivative


@pytest.mark.parametrize("degree", [2, 3, 5])
def test_nonuniform_derivative(degree):
    # Independent oracle: differentiate the ordinary spline forward twice in x.
    knots = torch.tensor([-4., -3.1, -2., -1.2, -.8, -.1, .4, 1.2, 2., 3., 4., 5., 6.],
                         dtype=torch.double, requires_grad=True)
    x = torch.tensor([[-.37], [.16], [.77]], dtype=torch.double, requires_grad=True)
    c = torch.randn(len(knots) - degree - 1, dtype=x.dtype, requires_grad=True)
    spline = (basis(x, knots, degree) * c).sum()
    first, = torch.autograd.grad(spline, x, create_graph=True)
    expected, = torch.autograd.grad(first.sum(), x, create_graph=True)
    actual = second_derivative(x, knots, c, degree)
    torch.testing.assert_close(actual, expected.squeeze(-1))
    for a, b in zip(torch.autograd.grad(actual.sum(), (c, knots)),
                    torch.autograd.grad(expected.sum(), (c, knots))):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_known_polynomial(degree):
    # Bezier control points for x^2 on [0,1] (affine for degrees 0 and 1).
    t = torch.tensor([0.] * (degree + 1) + [1.] * (degree + 1))
    c = torch.arange(degree + 1, dtype=t.dtype)
    if degree >= 2:
        c = c * (c - 1) / (degree * (degree - 1))
    x = torch.tensor([[.2], [.7]])
    actual = second_derivative(x, t, c, degree)
    torch.testing.assert_close(actual, torch.full_like(actual, 2 if degree >= 2 else 0))


def test_collision_and_entropy():
    for layer in (KANLinear(4, 2, groups=2), FixedLayer(4, 2)):
        with torch.no_grad():
            if isinstance(layer, KANLinear):
                layer.grid_bias.zero_()
            layer.grid[..., 3] = layer.grid[..., 2]
        x = layer.grid.flatten()[2].detach().expand(3, 4).clone().requires_grad_()
        if isinstance(layer, KANLinear):
            y, r = layer(x, return_curvature=True)
        else:
            layer.spline_weight.data.zero_()
            y, r = layer(x), layer.regularization_loss()
        assert torch.isfinite(y).all() and torch.isfinite(r)
        (y.sum() + r).backward()
        active = [p for name, p in layer.named_parameters() if name != "base_weight"]
        for tensor in [x, *active]:
            assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


def test_small_nonzero_span():
    a = torch.tensor([0., 1., 1e-12], requires_grad=True)
    b = torch.tensor([0., 0., 1e-12], requires_grad=True)
    result = divide_or_zero(a, b)
    torch.testing.assert_close(result, torch.tensor([0., 0., 1.]))
    result.sum().backward()
    assert a.grad[:2].tolist() == b.grad[:2].tolist() == [0., 0.]


@pytest.mark.parametrize("groups", [1, 2, -1])
def test_batch_reduction_and_training(groups):
    model = Knots_KAN([4, 4, 2], groups=groups).double()
    x = torch.randn(6, 4, dtype=torch.double)
    output, curvature = model(x, return_curvature=True, normalize=True)
    torch.testing.assert_close(output, model(x, normalize=True))
    h = (x - x.mean(1, keepdim=True)) / (x.var(1, keepdim=True, unbiased=False) + 1e-6).sqrt()
    expected = 0
    for norm, layer in zip(model.layer_norm, model.layers):
        u = norm(h)
        activations = (layer.b_splines(u) * layer.spline_lin_c).sum(-1)
        first, = torch.autograd.grad(activations.sum(), u, create_graph=True)
        second, = torch.autograd.grad(first.sum(), u, create_graph=True)
        expected += second.sum(1).mean()
        h = layer(u)
    torch.testing.assert_close(curvature, expected)
    (output.square().mean() + .01 * curvature).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    torch.optim.SGD(model.parameters(), lr=.001).step()


def test_reporting_ratios():
    import numpy as np
    from ts_utils.metrics import CORR, MAPE, MSPE, ND, RSE, SMAPE

    z = np.zeros((3, 1))
    with np.errstate(all="raise"):
        for metric in [MAPE, MSPE, ND, RSE]:
            assert metric(z, z) == 0
            assert np.isinf(metric(z + 1, z))
        assert SMAPE(z, z) == 0
        assert SMAPE(z + 1, z) == 200
        assert np.isnan(CORR(z, z))
        assert MAPE(z - 2e-8, z - 1e-8) == pytest.approx(100)


def test_rbf_width():
    from efficient_kan.rbf_kan import RadialBasisFunction

    with pytest.raises(ValueError):
        RadialBasisFunction(grid_min=0, grid_max=0)
    with pytest.raises(ValueError):
        RadialBasisFunction(denominator=0)
    assert RadialBasisFunction(num_grids=1, denominator=1)(torch.zeros(1)).isfinite().all()
