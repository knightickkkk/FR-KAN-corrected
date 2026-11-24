"""B-spline operations on the last axis, with broadcastable leading axes."""

import torch


def divide_or_zero(numerator, denominator):
    """Zero-width knot spans contribute zero, including during backpropagation."""
    valid = denominator != 0
    divisor = torch.where(valid, denominator, 1)
    return torch.where(valid, numerator / divisor, 0)


def basis(x, knots, degree):
    values = ((x >= knots[..., :-1]) & (x < knots[..., 1:])).to(x.dtype)
    for k in range(1, degree + 1):
        left = knots[..., :-(k + 1)]
        right = knots[..., k + 1:]
        a = divide_or_zero(x - left, knots[..., k:-1] - left)
        b = divide_or_zero(right - x, right - knots[..., 1:-k])
        values = a * values[..., :-1] + b * values[..., 1:]
    return values.contiguous()


def second_derivative(x, knots, coefficients, degree):
    """Evaluate the ordinary second derivative inside the selected knot span."""
    if degree < 2:
        return (coefficients.sum(-1) + x.squeeze(-1)) * 0
    # Keep the outer basis functions: FR-KAN also evaluates the padded support.
    values = basis(x, knots, degree - 2)
    for k in (degree - 1, degree):
        left_width = knots[..., k:-1] - knots[..., :-(k + 1)]
        right_width = knots[..., k + 1:] - knots[..., 1:-k]
        left = divide_or_zero(values[..., :-1], left_width)
        right = divide_or_zero(values[..., 1:], right_width)
        values = k * (left - right)
    return (values * coefficients).sum(-1)
