from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random
import os
from collections import defaultdict
from collections import OrderedDict

import json_tricks as json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints

from nms.nms import oks_nms
from nms.nms import soft_oks_nms

logger = logging.getLogger(__name__)

class PredictionDataset(Dataset):
    '''
    "keypoints": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    },
	"skeleton": [
        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    '''
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.nms_thre = cfg.TEST.NMS_THRE
        self.image_thre = cfg.TEST.IMAGE_THRE
        self.soft_nms = cfg.TEST.SOFT_NMS
        self.oks_thre = cfg.TEST.OKS_THRE
        self.in_vis_thre = cfg.TEST.IN_VIS_THRE

        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1
        
        self.yolo_version = cfg.YOLO.VERSION
        self.yolo_model_def = cfg.YOLO.MODEL_DEF
        self.yolo_class_path = cfg.YOLO.CLASS_PATH
        self.yolo_weights_path = cfg.YOLO.WEIGHTS
        self.yolo_batch_size = cfg.YOLO.BATCH_SIZE
        self.yolo_device = cfg.YOLO.DEVICE

        if self.yolo_version == 'v3':
            from models.detectors.YOLOv3 import YOLOv3
        elif self.yolo_version == 'v5':
            from models.detectors.YOLOv5 import YOLOv5
        else:
            raise ValueError('Unsopported YOLO version.')

        if self.yolo_version == 'v3':
            self.detector = YOLOv3(model_def=self.yolo_model_def,
                                    class_path=self.yolo_class_path,
                                    weights_path=self.yolo_weights_path,
                                    classes=('person',),
                                    max_batch_size=self.yolo_batch_size,
                                    device=self.yolo_device)
        else:
            self.detector = YOLOv5(model_def=self.yolo_model_def,
                                    device=self.yolo_device)

        self.transform = transform
        self.db = self._get_db()

    def _get_db(self):
        image_extensions = ('.jpg', '.png')  # Supported extensions
        image_path = os.path.join(self.root,self.image_set)
        image_names = [file for file in os.listdir(image_path) if file.lower().endswith(image_extensions)]
        db = []
        for impath in image_names:
            impath = os.path.join(self.root,self.image_set,impath)
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float32)
            joints_3d_vis = np.ones((self.num_joints, 3), dtype=np.float32)
            
            im = cv2.imread(impath, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            detections = self.detector.predict_single(im)
            nof_people = len(detections) if detections is not None else 0
            if nof_people != 1:
                raise ValueError('More than 1 human detected or no humans detected')
            
            x1, y1, x2, y2, conf, cls_conf, cls_pred = detections[0]
            x1 = int(round(x1.item()))
            x2 = int(round(x2.item()))
            y1 = int(round(y1.item()))
            y2 = int(round(y2.item()))
            w = x2 - x1
            h = y2 - y1

            center, scale = self._xywh2cs(x1,y1,w,h)

            db.append({
                'image': impath,
                'center': center,
                'scale': scale,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
            })
        logger.info(f'=> Total images loaded: {len(db)}')
        return db
    
    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + (w - 1) * 0.5
        center[1] = y + (h - 1) * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def evaluate(self, cfg, preds, output_dir, all_boxes, img_path,
                 *args, **kwargs):
        rank = cfg.RANK

        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            try:
                os.makedirs(res_folder)
            except Exception:
                logger.error('Fail to make {}'.format(res_folder))

        res_file = os.path.join(
            res_folder, 'keypoints_{}_results_{}.json'.format(
                self.image_set, rank)
        )

        # person x (keypoints)
        _kpts = []
        for idx, kpt in enumerate(preds):
            _img = img_path[idx].split('/')[-1][:-4]
            
            _kpts.append({
                'keypoints': kpt,
                'center': all_boxes[idx][0:2],
                'scale': all_boxes[idx][2:4],
                'area': all_boxes[idx][4],
                'score': all_boxes[idx][5],
                'image': _img
            })
        # image x person x (keypoints)
        kpts = defaultdict(list)
        for kpt in _kpts:
            kpts[kpt['image']].append(kpt)

        # rescoring and oks nms
        num_joints = self.num_joints
        in_vis_thre = self.in_vis_thre
        oks_thre = self.oks_thre
        oks_nmsed_kpts = []
        for img in kpts.keys():
            img_kpts = kpts[img]
            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p['keypoints'][n_jt][2]
                    if t_s > in_vis_thre:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = kpt_score * box_score

            if self.soft_nms:
                keep = soft_oks_nms(
                    [img_kpts[i] for i in range(len(img_kpts))],
                    oks_thre
                )
            else:
                keep = oks_nms(
                    [img_kpts[i] for i in range(len(img_kpts))],
                    oks_thre
                )

            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_keypoint_results(oks_nmsed_kpts, res_file)

    
    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        data_numpy = cv2.imread(
            image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        
        #print(image_file)
        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        if 'synthetic' in db_rec:
            syn = db_rec['synthetic']
        else:
            syn = False
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        # if self.is_train:
        #     if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
        #         and np.random.rand() < self.prob_half_body):
        #         c_half_body, s_half_body = self.half_body_transform(
        #             joints, joints_vis
        #         )

        #         if c_half_body is not None and s_half_body is not None:
        #             c, s = c_half_body, s_half_body

        #     sf = self.scale_factor
        #     rf = self.rotation_factor
        #     s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
        #     r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
        #         if random.random() <= 0.6 else 0

        #     if self.flip and random.random() <= 0.5:
        #         data_numpy = data_numpy[:, ::-1, :]
        #         joints, joints_vis = fliplr_joints(
        #             joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
        #         c[0] = data_numpy.shape[1] - c[0] - 1
                
        joints_heatmap = joints.copy()
        trans = get_affine_transform(c, s, r, self.image_size)
        trans_heatmap = get_affine_transform(c, s, r, self.heatmap_size)

        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
                joints_heatmap[i, 0:2] = affine_transform(joints_heatmap[i, 0:2], trans_heatmap)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score,
            'synthetic': syn
        }

        return input, meta
    
    def _write_keypoint_results(self, keypoints, res_file):
        results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array([img_kpts[k]['keypoints']
                                    for k in range(len(img_kpts))])
            key_points = np.zeros(
                (_key_points.shape[0], self.num_joints * 3), dtype=np.float32
            )

            for ipt in range(self.num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # keypoints score.
            
            key_points = self._remapToOpenPose(key_points)

            result = [
                {
                    'image_id': img_kpts[k]['image'],
                    'keypoints': list(key_points[k]),
                    'score': img_kpts[k]['score'],
                    'center': list(img_kpts[k]['center']),
                    'scale': list(img_kpts[k]['scale'])
                }
                for k in range(len(img_kpts))
            ]
            results.extend(result)
        logger.info('=> writing results json to %s' % res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)
    
    def _remapToOpenPose(self, keypoints):
        joints = keypoints.reshape(-1,3)
        coco_joints = {
            0: "nose",
            1: "left_eye",
            2: "right_eye",
            3: "left_ear",
            4: "right_ear",
            5: "left_shoulder",
            6: "right_shoulder",
            7: "left_elbow",
            8: "right_elbow",
            9: "left_wrist",
            10: "right_wrist",
            11: "left_hip",
            12: "right_hip",
            13: "left_knee",
            14: "right_knee",
            15: "left_ankle",
            16: "right_ankle",
            17: "neck",  # computed value, not in original coco format
            18: "mid_hip" # computed value, not in original coco format
        }
        op_joints ={
            0:  "nose",
            1:  "neck",
            2:  "right_shoulder",
            3:  "right_elbow",
            4:  "right_wrist",
            5:  "left_shoulder",
            6:  "left_elbow",
            7:  "left_wrist",
            8:  "mid_hip",
            9:  "right_hip",
            10: "right_knee",
            11: "right_ankle",
            12: "left_hip",
            13: "left_knee",
            14: "left_ankle",
            15: "right_eye",
            16: "left_eye",
            17: "right_ear",
            18: "left_ear",
            # 19: "background" # not used
        }

        # compute neck and mid_hip
        neck = (joints[5] + joints[6]) / 2.0
        mid_hip = (joints[11] + joints[12]) / 2.0
        joints = np.vstack((joints,neck,mid_hip))

        # remap from coco to openpose
        coco_to_op_map = []
        for op_idx, op_joint in op_joints.items():
            # Find the corresponding COCO index
            coco_idx = next((coco_idx for coco_idx, coco_joint in coco_joints.items() if coco_joint == op_joint), None)
            coco_to_op_map.append(coco_idx)
        op_joints = np.zeros((19,3))
        for op_idx, coco_idx in enumerate(coco_to_op_map):
                if coco_idx is not None:
                    op_joints[op_idx,:] = joints[coco_idx,:]

        return np.expand_dims(op_joints.flatten(),0)

