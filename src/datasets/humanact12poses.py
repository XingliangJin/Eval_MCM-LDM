import pickle as pkl
import numpy as np
import os
from .dataset import Dataset


import codecs as cs
import os
import random
from os.path import join as pjoin

import numpy as np
import spacy
import torch
from rich.progress import track
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm


def removetrans(motion_joint):
        # 去除轨迹
    root = motion_joint[:, 0, :]
    root = np.expand_dims(root,1).repeat(21,axis=1)
    motion_joint[:, 1:,:] = motion_joint[:,1:,:]-root
    motion_joint[:, 0, :] = 0.0
    return motion_joint




class HumanAct12Poses(Dataset):
    dataname = "humanact12"

    def __init__(self, datapath="data/HumanAct12Poses", **kargs):
        self.datapath = datapath

        super().__init__(**kargs)

        pkldatafilepath = os.path.join(datapath, "humanact12poses.pkl")
        data = pkl.load(open(pkldatafilepath, "rb"))

        self._pose = [x for x in data["poses"]]
        self._num_frames_in_video = [p.shape[0] for p in self._pose]
        self._joints = [x for x in data["joints3D"]]

        self._actions = [x for x in data["y"]]

        total_num_actions = 12
        self.num_classes = total_num_actions

        self._train = list(range(len(self._pose)))

        keep_actions = np.arange(0, total_num_actions)

        self._action_to_label = {x: i for i, x in enumerate(keep_actions)}
        self._label_to_action = {i: x for i, x in enumerate(keep_actions)}

        self._action_classes = humanact12_coarse_action_enumerator

    def _load_joints3D(self, ind, frame_ix):
        return self._joints[ind][frame_ix]

    def _load_rotvec(self, ind, frame_ix):
        pose = self._pose[ind][frame_ix].reshape(-1, 24, 3)
        return pose


humanact12_coarse_action_enumerator = {
    0: "warm_up",
    1: "walk",
    2: "run",
    3: "jump",
    4: "drink",
    5: "lift_dumbbell",
    6: "sit",
    7: "eat",
    8: "turn steering wheel",
    9: "phone",
    10: "boxing",
    11: "throw",
}


# for train sra 去掉轨迹 humanml3d_cmu_style
class humanml3d_cmu_style(Dataset):
    dataname = "humanml3d_cmu_style"

    def __init__(self,  datapath="dataset/label_style_CMU/gt_joints", **kargs):
        self.datapath = datapath

        super().__init__(**kargs)

        split_list = os.listdir(datapath)
        split_list.sort()
        #motion_dir = os.path.join(datapath, "new_joints")
        motion_dir = datapath

        progress_bar=True
        if progress_bar:
            enumerator = enumerate(
                track(
                    split_list,
                    f"Loading eval data {motion_dir}",
                ))
        # else:
        #     enumerator = enumerate(id_list)
        count = 0
        bad_count = 0
        new_name_list = []
        length_list = []
        motion_data = []
        motion_joint_data = []
        style_label = []
        for i, name in enumerator:
            motion_path = pjoin(motion_dir, name)
            if not os.path.isfile(motion_path):
                continue
            motion_joint = np.load(motion_path)
            motion_joint = motion_joint[:,:22,:]
            
            # 去除轨迹
            root = motion_joint[:, 0, :]
            root = np.expand_dims(root,1).repeat(21,axis=1)
            motion_joint[:, 1:,:] = motion_joint[:,1:,:]-root
            motion_joint[:, 0, :] = 0.0

            if len(motion_joint.shape) == 2:
                continue
            motion = motion_joint.reshape(motion_joint.shape[0], 66)

            motion_data.append(motion)
            motion_joint_data.append(motion_joint)

            id = name.split('.')[0].split('-')[-1]
            style_label.append(int(id))
            # id = name.split('_')[1]# style id for gen
            # # id = name.split('.')[0]# style id for gt
            # with cs.open(pjoin(style_text_dir, id + ".txt")) as f:
            #     style_line = f.readline()
            #     label = style_line.strip()
                

            #     style_label.append(int(label))










        self._pose = motion_data
        self._num_frames_in_video = [p.shape[0] for p in self._pose]
        self._joints = motion_joint_data

        self._actions = style_label

        total_num_actions = 6
        self.num_classes = total_num_actions

        self._train = list(range(len(self._pose)))

        keep_actions = np.arange(0, total_num_actions)

        self._action_to_label = {x: i for i, x in enumerate(keep_actions)}
        self._label_to_action = {i: x for i, x in enumerate(keep_actions)}

        self._action_classes = humanml3d_cmu_style_coarse_action_enumerator

    def _load_joints3D(self, ind, frame_ix):
        return self._joints[ind][frame_ix]

    def _load_rotvec(self, ind, frame_ix):
        pose = self._pose[ind][frame_ix].reshape(-1, 22, 3)
        return pose

humanml3d_cmu_style_coarse_action_enumerator = {
    0: "childlike",
    1: "happy",
    2: "old",
    3: "joy",
    4: "benhit",
    5: "stealthy",
}




class humanml3d_style(Dataset):
    dataname = "humanml3d_style"

    def __init__(self,  *eval_motion_path,eval_type, datapath="", **kargs):
        self.datapath = datapath

        super().__init__(**kargs)

        pkldatafilepath = eval_motion_path[0]
        data = pkl.load(open(pkldatafilepath, "rb"))

        self._pose = [x.reshape(-1, 66) for x in data["joints"]]
        self._num_frames_in_video = [p.shape[0] for p in self._pose]
        self._joints = [removetrans(x) for x in data["joints"]]

        # self._actions = [int(x) for x in data["label_style"]]# for transferred style
        if eval_type == 'ori':
            self._actions = [int(x) for x in data["label_content"]]# for ori style
        else:
            self._actions = [int(x) for x in data["label_style"]]# for transferred style

        total_num_actions = 6
        self.num_classes = total_num_actions

        self._train = list(range(len(self._pose)))

        keep_actions = np.arange(0, total_num_actions)

        self._action_to_label = {x: i for i, x in enumerate(keep_actions)}
        self._label_to_action = {i: x for i, x in enumerate(keep_actions)}

        self._action_classes = humanml3d_cmu_style_coarse_action_enumerator

    def _load_joints3D(self, ind, frame_ix):
        return self._joints[ind][frame_ix]

    def _load_rotvec(self, ind, frame_ix):
        pose = self._pose[ind][frame_ix].reshape(-1, 22, 3)
        return pose

humanml3d_cmu_style_coarse_action_enumerator = {
    0: "childlike",
    1: "happy",
    2: "old",
    3: "joy",
    4: "benhit",
    5: "stealthy",
}

# for eval cra and fid
class humanml3d_gen(Dataset):
    dataname = "humanml3d_gen"

    def __init__(self, *eval_motion_path, datapath="data/HumanAct12Poses", **kargs):
        self.datapath = datapath

        super().__init__(**kargs)

        pkldatafilepath = eval_motion_path[0]
        data = pkl.load(open(pkldatafilepath, "rb"))

        self._pose = [x.reshape(-1, 66) for x in data["joints"]]
        self._num_frames_in_video = [p.shape[0] for p in self._pose]
        self._joints = [removetrans(x) for x in data["joints"]]
        self._actions = [int(x) for x in data["label_content"]]

        total_num_actions = 8
        self.num_classes = total_num_actions

        self._train = list(range(len(self._pose)))

        keep_actions = np.arange(0, total_num_actions)

        self._action_to_label = {x: i for i, x in enumerate(keep_actions)}
        self._label_to_action = {i: x for i, x in enumerate(keep_actions)}

        self._action_classes = humanml3d_cmu_coarse_action_enumerator

    def _load_joints3D(self, ind, frame_ix):
        return self._joints[ind][frame_ix]

    def _load_rotvec(self, ind, frame_ix):
        pose = self._pose[ind][frame_ix].reshape(-1, 22, 3)
        return pose

humanml3d_cmu_coarse_action_enumerator = {
    0: "Walk",
    1: "Wash",
    2: "Run",
    3: "Jump",
    4: "Animal Behavior",
    5: "Dance",
    6: "Step",
    7: "Climb",
}



# train cra dataset: humanml3d_cmu       # inference time: load gt datset
class humanml3d_cmu(Dataset):
    dataname = "humanml3d_cmu"

    def __init__(self, datapath="dataset/content_CMU", **kargs):
        self.datapath = datapath

        super().__init__(**kargs)

        # pkldatafilepath = os.path.join(datapath, "train.txt")
        # data = pkl.load(open(pkldatafilepath, "rb"))

        #split_file = os.path.join(datapath, "train.txt")
        split_list = os.listdir(os.path.join(datapath, "joints3d"))
        split_list.sort()
        #motion_dir = os.path.join(datapath, "new_joints")
        motion_dir = os.path.join(datapath, "joints3d")
        style_text_dir = os.path.join(datapath, "numeric_id_for_cmu_label.txt")

        id_list = []
        with cs.open(style_text_dir, "r") as f:
            for line in f.readlines():
                id_list.append(int(line.strip()))
        data_dict = {}
        # id_list = []

        # with cs.open(split_file, "r") as f:
        #     for line in f.readlines():
        #         id_list.append(line.strip())
        # for i in split_list:
        #     id = i.split('_')[1]# style id
        #     id_list.append(id)
        # self.id_list = id_list

        progress_bar=True
        if progress_bar:
            enumerator = enumerate(
                track(
                    split_list,
                    f"Loading eval data {motion_dir}",
                ))
        # else:
        #     enumerator = enumerate(id_list)
        count = 0
        bad_count = 0
        new_name_list = []
        length_list = []
        motion_data = []
        motion_joint_data = []
        style_label = []
        for i, name in enumerator:
            motion_path = pjoin(motion_dir, name)
            if not os.path.isfile(motion_path):
                continue
            motion_joint = np.load(motion_path)
            
            # 去除轨迹
            root = motion_joint[:, 0, :]
            root = np.expand_dims(root,1).repeat(21,axis=1)
            motion_joint[:, 1:,:] = motion_joint[:,1:,:]-root
            motion_joint[:, 0, :] = 0.0

            if len(motion_joint.shape) == 2:
                continue
            motion = motion_joint.reshape(motion_joint.shape[0], 66)

            motion_data.append(motion)
            motion_joint_data.append(motion_joint)
            # id = name.split('_')[1]# style id for gen
            # # id = name.split('.')[0]# style id for gt
            # with cs.open(pjoin(style_text_dir, id + ".txt")) as f:
            #     style_line = f.readline()
            #     label = style_line.strip()
                

            #     style_label.append(int(label))










        self._pose = motion_data
        self._num_frames_in_video = [p.shape[0] for p in self._pose]
        self._joints = motion_joint_data

        self._actions = id_list

        total_num_actions = 8
        self.num_classes = total_num_actions

        self._train = list(range(len(self._pose)))

        keep_actions = np.arange(0, total_num_actions)

        self._action_to_label = {x: i for i, x in enumerate(keep_actions)}
        self._label_to_action = {i: x for i, x in enumerate(keep_actions)}

        self._action_classes = humanml3d_cmu_coarse_action_enumerator

    def _load_joints3D(self, ind, frame_ix):
        return self._joints[ind][frame_ix]

    def _load_rotvec(self, ind, frame_ix):
        pose = self._pose[ind][frame_ix].reshape(-1, 22, 3)
        return pose


humanml3d_cmu_coarse_action_enumerator = {
    0: "Walk",
    1: "Wash",
    2: "Run",
    3: "Jump",
    4: "Animal Behavior",
    5: "Dance",
    6: "Step",
    7: "Climb",
}


class humanml3d_eval(Dataset):
    dataname = "humanml3d_cmu"

    def __init__(self, *eval_motion_path, datapath="dataset/content_CMU/gt_joints", **kargs):
        self.datapath = datapath

        super().__init__(**kargs)

        # pkldatafilepath = os.path.join(datapath, "train.txt")
        # data = pkl.load(open(pkldatafilepath, "rb"))

        #split_file = os.path.join(datapath, "train.txt")
        split_list = os.listdir(eval_motion_path[0])
        split_list.sort()
        #motion_dir = os.path.join(datapath, "new_joints")
        motion_dir = eval_motion_path[0]
        # style_text_dir = os.path.join(datapath, "numeric_id_for_cmu_label.txt")

        # id_list = []
        # with cs.open(style_text_dir, "r") as f:
        #     for line in f.readlines():
        #         id_list.append(int(line.strip()))
        data_dict = {}
        # id_list = []

        # with cs.open(split_file, "r") as f:
        #     for line in f.readlines():
        #         id_list.append(line.strip())
        # for i in split_list:
        #     id = i.split('_')[1]# style id
        #     id_list.append(id)
        # self.id_list = id_list

        progress_bar=True
        if progress_bar:
            enumerator = enumerate(
                track(
                    split_list,
                    f"Loading eval data {motion_dir}",
                ))
        # else:
        #     enumerator = enumerate(id_list)
        count = 0
        bad_count = 0
        new_name_list = []
        length_list = []
        motion_data = []
        motion_joint_data = []
        style_label = []
        for i, name in enumerator:
            motion_path = pjoin(motion_dir, name)
            if not os.path.isfile(motion_path):
                continue
            motion_joint = np.load(motion_path)
            if len(motion_joint.shape) == 2:
                continue
            motion = motion_joint.reshape(motion_joint.shape[0], 66)

            motion_data.append(motion)
            motion_joint_data.append(motion_joint)
            id = name.split('.')[0].split('-')[-1] # for gt and motionpuzzle
            # id = name.split('_')[0].split('-')[-1] # for ours
            style_label.append(int(id))
            # id = name.split('_')[1]# style id for gen
            # # id = name.split('.')[0]# style id for gt
            # with cs.open(pjoin(style_text_dir, id + ".txt")) as f:
            #     style_line = f.readline()
            #     label = style_line.strip()
                

            #     style_label.append(int(label))










        self._pose = motion_data
        self._num_frames_in_video = [p.shape[0] for p in self._pose]
        self._joints = motion_joint_data

        self._actions = style_label

        total_num_actions = 8
        self.num_classes = total_num_actions

        self._train = list(range(len(self._pose)))

        keep_actions = np.arange(0, total_num_actions)

        self._action_to_label = {x: i for i, x in enumerate(keep_actions)}
        self._label_to_action = {i: x for i, x in enumerate(keep_actions)}

        self._action_classes = humanml3d_cmu_coarse_action_enumerator

    def _load_joints3D(self, ind, frame_ix):
        return self._joints[ind][frame_ix]

    def _load_rotvec(self, ind, frame_ix):
        pose = self._pose[ind][frame_ix].reshape(-1, 22, 3)
        return pose


humanml3d_cmu_coarse_action_enumerator = {
    0: "Walk",
    1: "Wash",
    2: "Run",
    3: "Jump",
    4: "Animal Behavior",
    5: "Dance",
    6: "Step",
    7: "Climb",
}

class humanml3d_gt(Dataset):
    dataname = "humanml3d_gt"

    def __init__(self, *eval_motion_path, datapath="dataset/content_CMU", **kargs):
        self.datapath = datapath

        super().__init__(**kargs)

        # pkldatafilepath = os.path.join(datapath, "train.txt")
        # data = pkl.load(open(pkldatafilepath, "rb"))

        #split_file = os.path.join(datapath, "train.txt")
        split_list = os.listdir(eval_motion_path[0])
        split_list.sort()
        #motion_dir = os.path.join(datapath, "new_joints")
        motion_dir = eval_motion_path[0]
        # style_text_dir = os.path.join(datapath, "numeric_id_for_cmu_label.txt")

        # id_list = []
        # with cs.open(style_text_dir, "r") as f:
        #     for line in f.readlines():
        #         id_list.append(int(line.strip()))
        data_dict = {}
        # id_list = []

        # with cs.open(split_file, "r") as f:
        #     for line in f.readlines():
        #         id_list.append(line.strip())
        # for i in split_list:
        #     id = i.split('_')[1]# style id
        #     id_list.append(id)
        # self.id_list = id_list

        progress_bar=True
        if progress_bar:
            enumerator = enumerate(
                track(
                    split_list,
                    f"Loading eval data {motion_dir}",
                ))
        # else:
        #     enumerator = enumerate(id_list)
        count = 0
        bad_count = 0
        new_name_list = []
        length_list = []
        motion_data = []
        motion_joint_data = []
        style_label = []
        for i, name in enumerator:
            motion_path = pjoin(motion_dir, name)
            if not os.path.isfile(motion_path):
                continue
            motion_joint = np.load(motion_path)
            if len(motion_joint.shape) == 2:
                continue
            motion = motion_joint.reshape(motion_joint.shape[0], 66)

            motion_data.append(motion)
            motion_joint_data.append(motion_joint)
            id = name.split('.')[0].split('-')[-1] # for gt and motionpuzzle
            # id = name.split('_')[0].split('-')[-1] # for ours
            style_label.append(int(id))
            # id = name.split('_')[1]# style id for gen
            # # id = name.split('.')[0]# style id for gt
            # with cs.open(pjoin(style_text_dir, id + ".txt")) as f:
            #     style_line = f.readline()
            #     label = style_line.strip()
                

            #     style_label.append(int(label))










        self._pose = motion_data
        self._num_frames_in_video = [p.shape[0] for p in self._pose]
        self._joints = motion_joint_data

        self._actions = style_label

        total_num_actions = 8
        self.num_classes = total_num_actions

        self._train = list(range(len(self._pose)))

        keep_actions = np.arange(0, total_num_actions)

        self._action_to_label = {x: i for i, x in enumerate(keep_actions)}
        self._label_to_action = {i: x for i, x in enumerate(keep_actions)}

        self._action_classes = humanml3d_cmu_coarse_action_enumerator

    def _load_joints3D(self, ind, frame_ix):
        return self._joints[ind][frame_ix]

    def _load_rotvec(self, ind, frame_ix):
        pose = self._pose[ind][frame_ix].reshape(-1, 22, 3)
        return pose


humanml3d_cmu_coarse_action_enumerator = {
    0: "Walk",
    1: "Wash",
    2: "Run",
    3: "Jump",
    4: "Animal Behavior",
    5: "Dance",
    6: "Step",
    7: "Climb",
}
