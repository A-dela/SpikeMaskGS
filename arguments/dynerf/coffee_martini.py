_base_ = './default.py'
OptimizationParams = dict(
    activation = "None",
    custom_sampler = None, # review random may be better
)

PipelineParams = dict(
    dsh=True # review
)

ModelHiddenParams = dict(
    no_do=False, # review default:False, set to True might cause floating static points
    no_dshs=False,
)