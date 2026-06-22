_base_ = './default.py'
OptimizationParams = dict(
    activation = "None",
    hashmap_size = 19,
    hash_final_lr=0.00002,
    coarse_iterations=6000,
    iterations = 23_000,
    custom_sampler = True
)
PipelineParams = dict(
    binary = True,
    mask_path = "rc_mask_80",
    sigmoid_binary = True,  
    # sigmoid_binary = False,    
    dsh=False
)