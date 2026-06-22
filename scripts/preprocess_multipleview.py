from argparse import ArgumentParser
import sys
sys.path.append('.')
from scene.nvidia_dataset_NDC import NVIDIA_NDC_Dataset
from scene.multipleview_dataset import multipleview_dataset
import os
from scene.dataset_readers import read_extrinsics_binary, read_intrinsics_binary
# import scene
# from scene.neural_3D_dataset_NDC import Neural3D_NDC_Dataset

if __name__ == '__main__':
    parser = ArgumentParser(description="Extract images from dynerf videos")
    parser.add_argument("--datadir", default='data/dynerf/cut_roasted_beef', type=str)
    args = parser.parse_args()
    # train_dataset = NVIDIA_NDC_Dataset(args.datadir, "train", 1.0, time_scale=1, 
    #                                      scene_bbox_min=[-2.5, -2.0, -1.0], scene_bbox_max=[2.5, 2.0, 1.0], eval_index=0)    
    # test_dataset = NVIDIA_NDC_Dataset(args.datadir, "test", 1.0, time_scale=1, 
    #                                     scene_bbox_min=[-2.5, -2.0, -1.0], scene_bbox_max=[2.5, 2.0, 1.0], eval_index=0)
    cameras_extrinsic_file = os.path.join(args.datadir, "sparse_/images.bin")
    cameras_intrinsic_file = os.path.join(args.datadir, "sparse_/cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    from scene.multipleview_dataset import multipleview_dataset
    train_cam_infos = multipleview_dataset(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, cam_folder=args.datadir,split="train")
    test_cam_infos = multipleview_dataset(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, cam_folder=args.datadir,split="test")
