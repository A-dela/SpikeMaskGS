#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_short
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams,OptimizationParams 
from gaussian_renderer import GaussianModel
from time import time
# import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import threading
import concurrent.futures
test_sigmoid=False
def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
def render_set(hyper, model_path, name, iteration, views, gaussians, pipeline, background, cam_type, residual=False, dyn_mask=False, real_dyn_mask=False, real_dyn_label_mask=False, dyn_image=False, short=False, pts=False, dyn_map=False, render_pixel=False, mask_map=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    residual_path = os.path.join(model_path, name, "ours_{}".format(iteration), "residual")
    dyn_mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "dyn_mask")
    real_dyn_mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "real_dyn_mask")
    real_dyn_label_mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "real_dyn_label_mask")
    dyn_image_path = os.path.join(model_path, name, "ours_{}".format(iteration), "dyn_image")
    sta_image_path = os.path.join(model_path, name, "ours_{}".format(iteration), "sta_image")
    dy_pts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "dy_pts_image")
    st_pts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "st_pts_image")
    all_pts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "all_pts_image")
    dy_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "dy_map_image")
    render_pixel_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_pixel")
    hist_pixel_path = os.path.join(model_path, name, "ours_{}".format(iteration), "hist_pixel")
    mask_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "mask_map")
    
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_images = []
    gt_list = []
    if residual:
        residual_list = []
        makedirs(residual_path, exist_ok=True)
    if dyn_mask:
        makedirs(dyn_mask_path, exist_ok=True)
        dyn_mask_list = []
    if real_dyn_mask:
        makedirs(real_dyn_mask_path, exist_ok=True)
        real_dyn_mask_list = []
    if real_dyn_label_mask:
        makedirs(real_dyn_label_mask_path, exist_ok=True)
        real_dyn_label_mask_list = []
    if dyn_image:
        makedirs(dyn_image_path, exist_ok=True)
        makedirs(sta_image_path, exist_ok=True)
        dyn_image_list = []
        sta_image_list = []
    if pts:
        makedirs(dy_pts_path, exist_ok=True)
        makedirs(st_pts_path, exist_ok=True)
        makedirs(all_pts_path, exist_ok=True)
        dy_pts_list = []
        st_pts_list = []
        all_pts_list = []
    if dyn_map:
        dyn_map_list = []
        makedirs(dy_map_path, exist_ok=True)
    if render_pixel:
        render_pixel_list = []
        hist_pixel_list = []
        makedirs(render_pixel_path, exist_ok=True)
        makedirs(hist_pixel_path, exist_ok=True)
    if mask_map:
        mask_map_list = []
        makedirs(mask_map_path, exist_ok=True)
    render_list = []
    # breakpoint()
    print("point nums:",gaussians._xyz.shape[0])
    if test_sigmoid:
        dynamics = ( gaussians.get_binary_sigmoid > 0.70 )
        print("The num of dynamic point: {}".format(torch.sum(dynamics)))
    elif pipeline.binary:
        dynamics = gaussians.get_binary_dynamic
        if hyper.STE:
            dynamics = gaussians.get_binary_ste
        elif hyper.Gumbel:
            dynamics = gaussians.get_binary_gumbel
        elif pipeline.sigmoid_binary:
            dynamics = gaussians.get_binary_dynamic_sigmoid
        print("The num of dynamic point: {}".format(torch.sum(dynamics)))
    else:
        print("The num of dynamic point: {}".format(torch.sum(gaussians.get_dynamic > gaussians.dynamic_thro)))
    all_time = 0

    numbers = np.linspace(20, -20, 150)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        start = 0
        end =2
        # rendering = render(view, gaussians, pipeline, background, cam_type=cam_type, stage= "coarse")["render"]
        if (name == "train" and idx>=start and idx<end) or name=="test":
            time1 = time()  
            if not short:
                # rendering = render(view, gaussians, pipeline, background, cam_type=cam_type, stage= "fine", mode="render", STE=hyper.STE, Gumbel=hyper.Gumbel)["render"]
                render_pkg=render(view, gaussians, pipeline, background, cam_type=cam_type, stage= "fine", mode="render", STE=hyper.STE, Gumbel=hyper.Gumbel)
                rendering, viewspace_point_tensor, visibility_filter, radii, max_weight_t, rendering_pixel, mask_map1 = \
                        render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"],render_pkg["max_weight_t"],render_pkg["rendering_pixel"],render_pkg["mask_map"]
                if test_sigmoid:
                    dynamics = gaussians.get_binary_sigmoid
                    # dynamics = gaussians.get_binary_sigmoid > 0.70
                elif pipeline.binary:
                    dynamics = gaussians.get_binary_dynamic
                    if hyper.STE:
                        dynamics = gaussians.get_binary_ste
                    elif hyper.Gumbel:
                        dynamics = gaussians.get_binary_gumbel
                    elif pipeline.sigmoid_binary:
                        dynamics = gaussians.get_binary_dynamic_sigmoid
                else:
                    dynamics = gaussians.get_dynamic
                # pixel_gs = dynamics[max_weight_t.bool()]
                # pixel_gs = gaussians.get_dynamic[max_weight_t.bool()]
                pixel_gs = gaussians.get_binary_sigmoid[max_weight_t.bool()]
            else:
                rendering = render_short(view, gaussians, pipeline, background, cam_type=cam_type, stage= "fine", mode="render", STE=hyper.STE, Gumbel=hyper.Gumbel)["render"]
            time2 = time()
            all_time += (time2-time1)

            render_images.append(to8b(rendering).transpose(1,2,0))
            render_list.append(rendering)
            if name in ["train", "test"]:
                if cam_type != "PanopticSports":
                    gt = view.original_image[0:3, :, :]
                else:
                    gt  = view['image'].cuda()
                gt_list.append(gt)
            if residual:
                residual = torch.abs(rendering.cuda() - gt.cuda())
                residual_list.append(residual)
            if dyn_mask:
                dyn_map1= render(view, gaussians, pipeline, background, cam_type=cam_type, stage= "fine", mode="render", STE=hyper.STE, Gumbel=hyper.Gumbel)["dynamic_map"]
                dyn_mask_list.append(dyn_map1)
            if real_dyn_mask:
                rendered_dyn_mask = render(view, gaussians, pipeline, background, cam_type=cam_type, stage= "fine", mode="render", STE=hyper.STE, Gumbel=hyper.Gumbel)["dyn_map"]
                real_dyn_mask_list.append(rendered_dyn_mask)
            if real_dyn_label_mask:
                rendered_real_dyn_label_mask = render(view, gaussians, pipeline, background, cam_type=cam_type, stage= "fine", mode="render", STE=hyper.STE, Gumbel=hyper.Gumbel)["d_map"]
                real_dyn_label_mask_list.append(rendered_real_dyn_label_mask)
            if dyn_image:
                rendered_dyn_image = render(view, gaussians, pipeline, background, cam_type=cam_type, stage= "fine", mode="render", STE=hyper.STE, Gumbel=hyper.Gumbel)["dyn_image"]
                # rendered_dyn_image = view.dyn_image.cuda()
                rendered_sta_image = render(view, gaussians, pipeline, background, cam_type=cam_type, stage= "fine", mode="render", STE=hyper.STE, Gumbel=hyper.Gumbel)["static_image"]
                dyn_image_list.append(rendered_dyn_image)
                sta_image_list.append(rendered_sta_image)
            if pts:
                all_points = torch.clamp(render(view, gaussians, pipeline, background, cam_type=cam_type, stage= "fine", mode="render", STE=hyper.STE, Gumbel=hyper.Gumbel)["all_point"], 0.0, 1.0)
                dy_all_point = torch.clamp(render(view, gaussians, pipeline, background, cam_type=cam_type, stage= "fine", mode="render", STE=hyper.STE, Gumbel=hyper.Gumbel)["dy_all_point"], 0.0, 1.0)
                sta_all_point = torch.clamp(render(view, gaussians, pipeline, background, cam_type=cam_type, stage= "fine", mode="render", STE=hyper.STE, Gumbel=hyper.Gumbel)["sta_all_point"], 0.0, 1.0)
                dy_pts_list.append(dy_all_point)
                st_pts_list.append(sta_all_point)
                all_pts_list.append(all_points)
            if dyn_map:
                dy_map = torch.clamp(render(view, gaussians, pipeline, background, cam_type=cam_type, stage= "fine", mode="render", STE=hyper.STE, Gumbel=hyper.Gumbel)["dynamic_map"], 0.0, 1.0)
                dyn_map_list.append(dy_map)
            if mask_map:
                mask_map_list.append(mask_map1)
            if render_pixel:
                render_pixel_list.append(rendering_pixel)
                
                import seaborn as sns
                # plt.hist(pixel_gs.cpu().detach().numpy(), bins=30, density=True)
                # , legend=False
                # sns.kdeplot(pixel_gs.cpu().detach().numpy(),clip=(0,1),bw_adjust=0.01, legend=False)
                # sns.displot(
                #     data=pixel_gs.cpu().detach().numpy(), stat="percent", kde=False,
                # )
                plt.figure(figsize=(6, 3))
                ax=sns.histplot(pixel_gs.cpu().detach().numpy(), bins=50, stat="proportion", kde=False, legend=False)
                plt.ylim(0, 0.5)
                buf = BytesIO()
                plt.tick_params(axis='both', which='major', labelsize=14)
                # plt.title('d Distribution',  font={'size':14})
                plt.ylabel('Proportion',  font={'size':14})
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                image = Image.open(buf).convert('RGB')  # 转为PIL图像
                tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0 
                hist_pixel_list.append(tensor)
                plt.close()
        elif idx>=end:
            break
        else:
            continue
    # time2=time()
    print("FPS:",(len(views)-1)/all_time)

    print("writing  training  images.")
    multithread_write(gt_list, gts_path)
    print("writing  rendering  images.")
    multithread_write(render_list, render_path)
    if residual:
        print("writing  residual  images.")
        multithread_write(residual_list, residual_path)
    if dyn_mask:
        print("writing  dynamic mask  images.")
        multithread_write(dyn_mask_list, dyn_mask_path)
    if real_dyn_mask:
        print("writing  real  dynamic mask  images.")
        multithread_write(real_dyn_mask_list, real_dyn_mask_path)
    if real_dyn_label_mask:
        print("writing  real  dynamic label mask  images.")
        multithread_write(real_dyn_label_mask_list, real_dyn_label_mask_path)
    if dyn_image:
        print("writing  dynamic image  images.")
        multithread_write(dyn_image_list, dyn_image_path)
        print("writing  static image  images.")
        multithread_write(sta_image_list, sta_image_path)
    if pts:
        print("writing  dynamic pts  images.")
        multithread_write(dy_pts_list, dy_pts_path)
        print("writing  static pts  images.")
        multithread_write(st_pts_list, st_pts_path)
        print("writing  all pts  images.")
        multithread_write(all_pts_list, all_pts_path)
    if dyn_map:
        print("writing  dynamic map  images.")
        multithread_write(dyn_map_list, dy_map_path)
    if mask_map:
        print("writing  mask map  images.")
        multithread_write(mask_map_list, mask_map_path)
    if render_pixel:
        print("writing  rendering pixel  images.")
        multithread_write(render_pixel_list, render_pixel_path)
        print("writing  histogram pixel  images.")
        multithread_write(hist_pixel_list, hist_pixel_path)
    
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), render_images,fps=30)


    
def render_sets( opt , dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool, residual=False, dyn_mask=False, real_dyn_mask=False, real_dyn_label_mask=False, dyn_image=False, short=False, pts=False, dyn_map=False, render_pixel=False, mask_map=False):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam, opt)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        cam_type=scene.dataset_type
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(hyperparam,dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,cam_type, residual=residual, dyn_mask=dyn_mask, real_dyn_mask=real_dyn_mask, real_dyn_label_mask=real_dyn_label_mask, dyn_image=dyn_image, short=short, pts=pts, dyn_map=dyn_map, render_pixel=render_pixel, mask_map=mask_map)

        if not skip_test:
            render_set(hyperparam,dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,cam_type, residual=residual, dyn_mask=dyn_mask, real_dyn_mask=real_dyn_mask, real_dyn_label_mask=real_dyn_label_mask, dyn_image=dyn_image, short=short, pts=pts, dyn_map=dyn_map, render_pixel=render_pixel, mask_map=mask_map)
        if not skip_video:
            render_set(hyperparam,dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,cam_type, residual=residual, dyn_mask=dyn_mask, real_dyn_mask=real_dyn_mask, real_dyn_label_mask=real_dyn_label_mask, dyn_image=dyn_image, short=short)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    op = OptimizationParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--dyn_mask", action="store_true")
    parser.add_argument("--real_dyn_mask", action="store_true")
    parser.add_argument("--real_dyn_label_mask", action="store_true")
    parser.add_argument("--dyn_image", action="store_true")
    parser.add_argument("--pts", action="store_true")
    parser.add_argument("--short", action="store_true")
    parser.add_argument("--dyn_map", action="store_true")
    parser.add_argument("--render_pixel", action="store_true")
    parser.add_argument("--mask_map", action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmengine as mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(op.extract(args),  model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video, args.residual, args.dyn_mask, args.real_dyn_mask, args.real_dyn_label_mask, args.dyn_image, args.short, args.pts, args.dyn_map, args.render_pixel, args.mask_map)
