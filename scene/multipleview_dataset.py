import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils.graphics_utils import focal2fov
from scene.colmap_loader import qvec2rotmat
from scene.dataset_readers import CameraInfo
# from swift4d.scene.neural_3D_dataset_NDC_old import get_spiral
from scene.neural_3D_dataset_NDC import get_spiral
from torchvision import transforms as T
import glob
import cv2
import torch

class multipleview_dataset(Dataset):
    def __init__(
        self,
        cam_extrinsics,
        cam_intrinsics,
        cam_folder,
        split,
        downsample=1.0,
    ):
        self.n_frame = len(os.listdir(os.path.join(cam_folder, "cam01")))
        self.img_wh = (
            int(960 / downsample),
            int(540 / downsample),)
        self.root_dir = cam_folder
        self.downsample = 2704 / self.img_wh[0]
        self.focal = [cam_intrinsics[1].params[0], cam_intrinsics[1].params[0]]
        height=cam_intrinsics[1].height
        width=cam_intrinsics[1].width
        self.FovY = focal2fov(self.focal[0], height)
        self.FovX = focal2fov(self.focal[0], width)
        self.transform = T.ToTensor()
        self.image_paths, self.image_poses, self.image_times= self.load_images_path(cam_folder, cam_extrinsics,cam_intrinsics,split)
        videos = glob.glob(os.path.join(self.root_dir, "cam*.mp4"))  # root_dir 是数据集
        videos = sorted(videos)
        self.std_frames = []
        image_length = len(os.listdir(os.path.join(cam_folder, "cam01"))) # 151
        self.poses = np.ones((int(len(self.image_paths)/image_length),1), dtype=np.float32) # no meaning
        # self.poses = np.ones([sum(1 for entry in os.listdir(cam_folder) if os.path.isdir(os.path.join(cam_folder, entry)) and entry.startswith("cam"))]) # just for convenience
        self.calc_std(videos, os.path.join(self.root_dir, f"stds_{int(self.downsample)}"))
        if split=="test":
            self.video_cam_infos=self.get_video_cam_infos(cam_folder, image_length)
        
    def calc_std(self, video_path_root, std_path_root, frame_start=0, n_frame=12):
        print(f"Calculating std frames for {len(video_path_root)} videos...")
        if os.path.exists(std_path_root):
            std_files = glob.glob(os.path.join(std_path_root, "*.npy"))
            std_files = sorted(std_files)

            for i, std_file in enumerate(std_files):
                std_frame = np.load(std_file)
                std_frame = torch.from_numpy(std_frame)
                self.std_frames.append(std_frame)
            return
        
        os.makedirs(std_path_root)
        # print(video_path_root)
        all_image_range = range(n_frame) # train:加载所有相机的全部帧            
        test_image_range = [all_image_range[0], all_image_range[int(n_frame/3)], all_image_range[int(n_frame*2/3)]]
        image_range = [i for i in range(n_frame) if i not in test_image_range] # exclude train views

        for i , video_path in enumerate(video_path_root):
            print(f"Processing video {i+1}/{len(video_path_root)}: {video_path}")
            images_path = video_path.split('.')[0]
            # frame_paths = glob.glob(os.path.join(os.path.join(images_path), "*.jpg"))
            frame_paths = sorted(glob.glob(os.path.join(images_path, "*.jpg")))
            frame_paths = [path for i, path in enumerate(frame_paths) if i in image_range]
            print(f"Found {len(frame_paths)} frames in {images_path}")
            frames = []
            for j , fp in enumerate(frame_paths):
                frame = Image.open( fp ).convert('RGB')
                frame = np.array(frame, dtype=np.float32) / 255.
                frames.append(frame)
            frame = np.stack(frames, axis=0)  # [300,h,w,3]
            std_map = frame.std(axis=0).mean(axis=-1)  # [h,w]
            std_map_blur = (cv2.GaussianBlur(std_map, (31, 31), 0)).astype(np.float32) # [1014,1352] 高斯模糊
            std_path = os.path.join(std_path_root, os.path.basename(video_path)+'_std.npy')

            self.std_frames.append(std_map_blur)
            np.save(std_path, std_map_blur)
    
    # def load_images_path(self, cam_folder, cam_extrinsics,cam_intrinsics,split):
    #     image_length = len(os.listdir(os.path.join(cam_folder,"cam01")))
    #     #len_cam=len(cam_extrinsics)
    #     image_paths=[]
    #     image_poses=[]
    #     image_times=[]
    #     for idx, key in enumerate(cam_extrinsics):
    #         extr = cam_extrinsics[key]
    #         R = np.transpose(qvec2rotmat(extr.qvec))
    #         T = np.array(extr.tvec)

    #         number = os.path.basename(extr.name)[5:-4]
    #         images_folder=os.path.join(cam_folder,"cam"+number.zfill(2))
    #         print(f"Loading images from {images_folder} for camera {number} with {key}")

    #         image_range=range(image_length)
    #         if split=="test":
    #             image_range = [image_range[0],image_range[int(image_length/3)],image_range[int(image_length*2/3)]]

    #         for i in image_range:    
    #             num=i+1
    #             image_path=os.path.join(images_folder,"frame_"+str(num).zfill(5)+".jpg")
    #             image_paths.append(image_path)
    #             image_poses.append((R,T))
    #             image_times.append(float(i/image_length))

    #     return image_paths, image_poses,image_times
    
    def load_images_path(self, cam_folder, cam_extrinsics, cam_intrinsics, split):
        image_length = len(os.listdir(os.path.join(cam_folder, "cam01")))
        image_paths = []
        image_poses = []
        image_times = []
        
        # 收集所有相机信息
        camera_info = []
        for idx, key in enumerate(cam_extrinsics):
            extr = cam_extrinsics[key]
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)
            
            number = os.path.basename(extr.name)[5:-4]
            images_folder = os.path.join(cam_folder, "cam" + number.zfill(2))
            camera_info.append((key, int(number), R, T, images_folder))
        
        # 按number排序
        sorted_cameras = sorted(camera_info, key=lambda x: x[1])
        
        # 处理排序后的相机
        for key, number, R, T, images_folder in sorted_cameras:
            # print(f"Loading images from {images_folder} for camera {number} with {key}")
            
            # image_range = range(image_length)
            # if split == "test":
            #     image_range = [image_range[0], image_range[int(image_length/3)], image_range[int(image_length*2/3)]]
            all_image_range = range(image_length) # train:加载所有相机的全部帧            
            test_image_range = [all_image_range[0], all_image_range[int(image_length/3)], all_image_range[int(image_length*2/3)]]
            if split == "test": # test:仅加载每个相机的关键帧（第 1 帧、1/3 处的帧、2/3 处的帧
                image_range = test_image_range
            else: # if exclude test views in train split
                image_range = [i for i in range(image_length) if i not in test_image_range]
            
            for i in image_range:    
                num = i + 1
                image_path = os.path.join(images_folder, "frame_" + str(num).zfill(5) + ".jpg")
                image_paths.append(image_path)
                image_poses.append((R, T))
                image_times.append(float(i/image_length))
        
        return image_paths, image_poses, image_times
    
    def get_video_cam_infos(self,datadir,image_length):
        poses_arr = np.load(os.path.join(datadir, "poses_bounds_multipleview.npy"))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
        near_fars = poses_arr[:, -2:]
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        N_views = image_length
        val_poses = get_spiral(poses, near_fars, N_views=N_views)

        cameras = []
        len_poses = len(val_poses)
        times = [i/len_poses for i in range(len_poses)]
        image = Image.open(self.image_paths[0])
        image = self.transform(image)

        for idx, p in enumerate(val_poses):
            image_path = None
            image_name = f"{idx}"
            time = times[idx]
            pose = np.eye(4)
            pose[:3,:] = p[:3,:]
            R = pose[:3,:3]
            R = - R
            R[:,0] = -R[:,0]
            T = -pose[:3,3].dot(R)
            FovX = self.FovX
            FovY = self.FovY
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                time = time, mask=None))
        return cameras
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self,index):

        img = Image.open(self.image_paths[index])
        img = img.resize(self.img_wh, Image.LANCZOS)
        img = self.transform(img)
        return img, self.image_poses[index], self.image_times[index] , self.std_frames[index//self.n_frame] 
    def load_pose(self,index):
        return self.image_poses[index]