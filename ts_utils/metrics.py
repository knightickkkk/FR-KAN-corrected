import numpy as np


def _normalized_error(error, scale):
    if scale == 0:
        return 0.0 if error == 0 else np.inf
    return error / scale


def RSE(pred, true):
    return _normalized_error(np.linalg.norm(true - pred), np.linalg.norm(true - true.mean()))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    # Correlation is undefined if either series is constant.
    result = np.full(np.shape(u), np.nan, dtype=np.result_type(u, float))
    return np.divide(u, d, out=result, where=d != 0).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def _relative_error(pred, true):
    delta = np.subtract(pred, true)
    # A perfect zero target contributes zero. A miss at zero is unbounded.
    out = np.where(delta == 0, 0.0, np.inf)
    return np.divide(delta, true, out=out, where=np.asarray(true) != 0)


def MAPE(pred, true):
    return np.mean(np.abs(100 * _relative_error(pred, true)))


def MSPE(pred, true):
    return np.mean(np.square(_relative_error(pred, true)))


def SMAPE(pred, true):
    scale = np.abs(pred) + np.abs(true)
    delta = 200 * np.abs(pred - true)
    out = np.zeros_like(delta, dtype=np.result_type(delta, float))
    return np.divide(delta, scale, out=out, where=scale != 0).mean()


def ND(pred, true):
    return _normalized_error(np.mean(np.abs(true - pred)), np.mean(np.abs(true)))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    smape = SMAPE(pred, true)
    nd = ND(pred, true)

    return mae, mse, rmse, mape, mspe, smape, nd
