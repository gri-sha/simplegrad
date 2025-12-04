import numpy as np


TOL = 1e-5


def compare2tensors(sg, pt, tol=TOL):
    """Compare simplegrad tensor values with pytorch tensor values"""
    sg_vals = sg.values if hasattr(sg, "values") else sg
    pt_vals = pt.detach().numpy()
    assert np.allclose(sg_vals, pt_vals, atol=tol), f"Values don't match: sg={sg_vals}, pt={pt_vals}"


def compare_grads(sg, pt, tol=TOL):
    """Compare gradients between simplegrad and pytorch tensors"""
    if sg.grad is not None and pt.grad is not None:
        assert np.allclose(sg.grad, pt.grad.numpy(), atol=tol), f"Gradients don't match: sg={sg.grad}, pt={pt.grad.numpy()}"
