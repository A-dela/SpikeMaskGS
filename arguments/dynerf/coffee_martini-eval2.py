_base_ = './default.py'
OptimizationParams = dict(
    activation = "None",
    custom_sampler = None, # review random may be better
)

# ModelHiddenParams = dict(
#     no_do=False, # review default:False
# )
# PipelineParams = dict(
#     sigmoid_binary = False,
# )

ModelParams = dict(
    eval_index=2
)