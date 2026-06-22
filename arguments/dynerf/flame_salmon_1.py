_base_ = './default.py'
OptimizationParams = dict(
    lambda_dssim = 0.0,  # better ?
    custom_sampler = None, 
)

PipelineParams = dict(
    dsh=True # review
)

ModelHiddenParams = dict(
    no_do=False,
    no_dshs=False,
    no_ds=False
)

