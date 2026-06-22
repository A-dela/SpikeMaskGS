_base_ = './default.py'
OptimizationParams = dict(
    # lambda_dssim = 0.0,  # better ?
    custom_sampler = None
)
ModelParams = dict(
    eval_index=2
)