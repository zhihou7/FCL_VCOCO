import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import nn as mynn
from core.config import cfg
import utils.net as net_utils
from utils.net import SpacialConv, Conv2d, ResidualConv, weighted_binary_cross_entropy_interaction
# from modeling.fast_rcnn_heads import roi_2mlp_head
import ipdb
from torch.autograd import Variable
import pdb
from utils.cbam import SpatialGate, CBAM, CBAM_ks3, ChannelGate
from modeling.roi_xfrom.roi_align.functions.roi_align import RoIAlignFunction

OBJ_NUM = 80
'''
An implementation of PMFNet.
'''

class PMFNet_Baseline(nn.Module):
    """
    Human Object Interaction.
    This module including Human-centric branch and Interaction branch of PMFNet.
    Holistic Part Attention initially
    directly estimate all attentions for each fine grained action
    """

    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()

        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.roi_size = roi_size
        # ToDo: cfg
        hidden_dim = cfg.VCOCO.MLP_HEAD_DIM
        print('.......hidden_dim of VCOCO HOI: ', hidden_dim)
        # num_action_classes = cfg.VCOCO.NUM_ACTION_CLASSES  # 26
        action_mask = np.array(cfg.VCOCO.ACTION_MASK).T
        interaction_num_action_classes = action_mask.sum().item()  # 24
        self.interaction_num_action_classes = interaction_num_action_classes

        self.human_fc1 = nn.Linear(dim_in * roi_size ** 2, hidden_dim)  # 512
        self.object_fc1 = nn.Linear(dim_in * roi_size ** 2, hidden_dim)  # 512
        self.union_fc1 = nn.Linear(dim_in * roi_size ** 2, hidden_dim)
        self.union_resolution = cfg.KRCNN.HEATMAP_SIZE if cfg.VCOCO.KEYPOINTS_ON else cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        interaction_fc1_dim_in = 3 * hidden_dim

        self.pose_fc3 = nn.Linear(3 * cfg.KRCNN.HEATMAP_SIZE ** 2, 512)
        self.pose_fc4 = nn.Linear(512, hidden_dim)

        interaction_fc1_dim_in += hidden_dim

        self.interaction_fc1 = nn.Linear(interaction_fc1_dim_in, hidden_dim)
        self.interaction_action_score = nn.Linear(hidden_dim, interaction_num_action_classes)


    def _init_weights(self):
        # Initialize human centric branch
        mynn.init.XavierFill(self.human_fc1.weight)
        init.constant_(self.human_fc1.bias, 0)
        # mynn.init.XavierFill(self.human_fc2.weight)
        # init.constant_(self.human_fc2.bias, 0)

        init.normal_(self.human_action_score.weight, std=0.01)
        init.constant_(self.human_action_score.bias, 0)
        init.normal_(self.human_action_bbox_pred.weight, std=0.001)
        init.constant_(self.human_action_bbox_pred.bias, 0)

        # Initialize interaction branch(object action score)
        mynn.init.XavierFill(self.interaction_fc1.weight)
        init.constant_(self.interaction_fc1.bias, 0)
        # mynn.init.XavierFill(self.interaction_fc2.weight)
        # init.constant_(self.interaction_fc2.bias, 0)

        init.normal_(self.interaction_action_score.weight, std=0.01)
        init.constant_(self.interaction_action_score.bias, 0)

    def detectron_weight_mapping(self):
        # hc is human centric branch
        # io is interaction branch object part
        detectron_weight_mapping = {
            'human_fc1.weight': 'hc_fc1_w',
            'human_fc1.bias': 'hc_fc1_b',
            'human_fc2.weight': 'hc_fc2_w',
            'human_fc2.bias': 'hc_fc2_b',
            'human_action_score.weight': 'hc_score_w',
            'human_action_score.bias': 'hc_score_b',

            'interaction_fc1.weight': 'inter_fc1_w',
            'interaction_fc1.bias': 'inter_fc1_b',
        }
        return detectron_weight_mapping, []

    def forward(self, x, hoi_blob):
        x_human = self.roi_xform(
            x, hoi_blob,
            blob_rois='human_boxes',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        x_object = self.roi_xform(
            x, hoi_blob,
            blob_rois='object_boxes',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        x_human = x_human.view(x_human.size(0), -1)
        x_object = x_object.view(x_object.size(0), -1)

        # get inds from numpy
        device_id = x_human.get_device()
        interaction_human_inds = torch.from_numpy(
            hoi_blob['interaction_human_inds']).long().cuda(device_id)
        interaction_object_inds = torch.from_numpy(
            hoi_blob['interaction_object_inds']).long().cuda(device_id)
        # human score and bbox predict
        x_human = F.relu(self.human_fc1(x_human), inplace=True)
        x_object = F.relu(self.object_fc1(x_object), inplace=True)
        x_interaction = torch.cat((x_human[interaction_human_inds], x_object[interaction_object_inds]), dim=1)

        x_union = self.roi_xform(
            x, hoi_blob,
            blob_rois='union_boxes',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            # resolution=self.union_resolution,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        x_union = x_union.view(x_union.size(0), -1)
        x_union = F.relu(self.union_fc1(x_union), inplace=True)
        x_interaction = torch.cat((x_interaction, x_union), dim=1)

        kps_pred = hoi_blob['poseconfig']
        if isinstance(kps_pred, np.ndarray):
            kps_pred = torch.from_numpy(kps_pred).cuda(device_id)
        # import ipdb
        # ipdb.set_trace()
        # poseconfig = kps_pred.view(kps_pred.size(0), -1)
        x_pose_line = kps_pred.view(kps_pred.size(0), -1)
        x_pose_line = F.relu(self.pose_fc3(x_pose_line), inplace=True)
        x_pose_line = F.relu(self.pose_fc4(x_pose_line), inplace=True)

        x_interaction = torch.cat((x_interaction, x_pose_line), dim=1)
        x_interaction = F.relu(self.interaction_fc1(x_interaction), inplace=True)

        interaction_action_score = self.interaction_action_score(x_interaction)
        hoi_blob['interaction_action_score'] = interaction_action_score
        hoi_blob['interaction_affinity_score']= torch.zeros((interaction_human_inds.shape[0], 1)).cuda(device_id)  ### 2 classisification score

        return hoi_blob

    @staticmethod
    def loss(hoi_blob):
        interaction_action_score = hoi_blob['interaction_action_score']
        device_id = interaction_action_score.get_device()

        interaction_action_labels = torch.from_numpy(hoi_blob['interaction_action_labels']).float().cuda(device_id)
        interaction_action_loss = F.binary_cross_entropy_with_logits(
            interaction_action_score, interaction_action_labels)
        # get interaction branch predict action accuracy
        interaction_action_preds = \
            (interaction_action_score.sigmoid() > cfg.VCOCO.ACTION_THRESH).type_as(interaction_action_labels)
        interaction_action_accuray_cls = interaction_action_preds.eq(interaction_action_labels).float().mean()
        interaction_affinity_loss = torch.zeros(interaction_action_loss.shape).cuda(device_id)
        interaction_affinity_cls = torch.zeros(interaction_action_accuray_cls.shape).cuda(device_id)
        return interaction_action_loss, interaction_affinity_loss, \
               interaction_action_accuray_cls, interaction_affinity_cls


def get_cooccurence_matrix_coco(verb_class_num=24, obj_class_num=80):
    # Noticeably, PMFNet predicts 24 clases.
    if verb_class_num == 24:
        set_list = [(0, 38), (1, 31), (1, 32), (2, 43), (2, 44), (2, 77), (3, 1), (3, 19), (3, 28), (3, 46), (3, 47),
                (3, 48), (3, 49), (3, 51), (3, 52), (3, 54), (3, 55), (3, 56), (4, 2), (4, 3), (4, 4), (4, 6), (4, 7),
                (4, 8), (4, 9), (4, 18), (4, 21), (5, 68), (6, 33), (7, 64), (8, 47), (8, 48), (8, 49), (8, 50),
                (8, 51), (8, 52), (8, 53), (8, 54), (8, 55), (8, 56), (9, 2), (9, 4), (9, 14), (9, 18), (9, 21),
                (9, 25), (9, 27), (9, 29), (9, 57), (9, 58), (9, 60), (9, 61), (9, 62), (9, 64), (10, 31), (10, 32),
                (10, 37), (10, 38), (11, 14), (11, 57), (11, 58), (11, 60), (11, 61), (12, 40), (12, 41), (12, 42),
                (12, 46), (13, 1), (13, 25), (13, 26), (13, 27), (13, 29), (13, 30), (13, 31), (13, 32), (13, 33),
                (13, 34), (13, 35), (13, 37), (13, 38), (13, 39), (13, 40), (13, 41), (13, 42), (13, 47), (13, 50),
                (13, 68), (13, 74), (13, 75), (13, 78), (14, 30), (14, 33), (15, 43), (15, 44), (15, 45), (16, 1),
                (16, 2), (16, 3), (16, 4), (16, 5), (16, 6), (16, 7), (16, 8), (16, 11), (16, 14), (16, 15), (16, 16),
                (16, 17), (16, 18), (16, 19), (16, 20), (16, 21), (16, 24), (16, 25), (16, 26), (16, 27), (16, 28),
                (16, 29), (16, 30), (16, 31), (16, 32), (16, 33), (16, 34), (16, 35), (16, 36), (16, 37), (16, 38),
                (16, 39), (16, 40), (16, 41), (16, 42), (16, 43), (16, 44), (16, 45), (16, 46), (16, 47), (16, 48),
                (16, 49), (16, 51), (16, 53), (16, 54), (16, 55), (16, 56), (16, 57), (16, 61), (16, 62), (16, 63),
                (16, 64), (16, 65), (16, 66), (16, 67), (16, 68), (16, 73), (16, 74), (16, 75), (16, 77), (17, 35),
                (17, 39), (18, 33), (19, 31), (19, 32), (20, 74), (21, 1), (21, 2), (21, 4), (21, 8), (21, 9), (21, 14),
                (21, 15), (21, 16), (21, 17), (21, 18), (21, 19), (21, 21), (21, 25), (21, 26), (21, 27), (21, 28),
                (21, 29), (21, 30), (21, 31), (21, 32), (21, 33), (21, 34), (21, 35), (21, 36), (21, 37), (21, 38),
                (21, 39), (21, 40), (21, 41), (21, 42), (21, 43), (21, 44), (21, 45), (21, 46), (21, 47), (21, 48),
                (21, 49), (21, 50), (21, 51), (21, 52), (21, 53), (21, 54), (21, 55), (21, 56), (21, 57), (21, 64),
                (21, 65), (21, 66), (21, 67), (21, 68), (21, 73), (21, 74), (21, 77), (21, 78), (21, 79), (21, 80),
                (22, 32), (22, 37), (23, 30), (23, 33)]
    else:
        return
    import pickle
    import numpy as np
    verb_to_HO_matrix = np.zeros((len(set_list), verb_class_num))
    for i in range(len(set_list)):
        item = set_list[i]
        verb_to_HO_matrix[i][item[0]] = 1
    verb_to_HO_matrix = np.transpose(verb_to_HO_matrix)

    obj_to_HO_matrix = np.zeros((len(set_list), obj_class_num))
    for i in range(len(set_list)):
        item = set_list[i]
        obj_to_HO_matrix[i][item[1] - 1] = 1
    obj_to_HO_matrix = np.transpose(obj_to_HO_matrix)
    return verb_to_HO_matrix, obj_to_HO_matrix


class PMFNet_Final(nn.Module):
    """
    add relative coordinate to parts
    Human Object Interaction.
    This module including Human-centric branch and Interaction branch of InteractNet.
    """
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()

        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.crop_size = cfg.VCOCO.PART_CROP_SIZE
        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        # ToDo: cfg
        hidden_dim = cfg.VCOCO.MLP_HEAD_DIM
        # num_action_classes = cfg.VCOCO.NUM_ACTION_CLASSES
        action_mask = np.array(cfg.VCOCO.ACTION_MASK).T
        interaction_num_action_classes = action_mask.sum().item()

        interaction_fc1_dim_in = 3 * hidden_dim
        self.part_num = 17

        self.pose_fc1 = nn.Linear((self.part_num+1) * 258 * self.crop_size ** 2, 1024)
        self.pose_fc2 = nn.Linear(1024, hidden_dim*2)
        interaction_fc1_dim_in += hidden_dim*2

        ## semantic attention
        self.mlp = nn.Sequential(
            nn.Linear(3*64*64, 64),
            nn.ReLU(),
            nn.Linear(64, self.part_num)
        )

        self.pose_fc3 = nn.Linear(3 * cfg.KRCNN.HEATMAP_SIZE ** 2, 512)
        self.pose_fc4 = nn.Linear(512, hidden_dim)
        interaction_fc1_dim_in += hidden_dim

        self.human_fc1 = nn.Linear(dim_in * roi_size ** 2, hidden_dim)
        self.object_fc1 = nn.Linear(dim_in * roi_size ** 2, hidden_dim)
        self.union_fc1 = nn.Linear(dim_in * roi_size ** 2, hidden_dim)

        self.interaction_fc1 = nn.Linear(interaction_fc1_dim_in, hidden_dim)
        self.interaction_action_score = nn.Linear(hidden_dim, interaction_num_action_classes)

        self.global_affinity = nn.Sequential(
            nn.Linear(interaction_fc1_dim_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        import os
        self.obj_embedding = torch.nn.Embedding(80, 256, scale_grad_by_freq=True)
        self.verb_to_HO_matrix, self.obj_to_HO_matrix = get_cooccurence_matrix_coco()
        self.obj_fabricator = nn.Sequential(
            nn.Linear(256+256+256, 512, bias=False),
            # nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256, bias=False),
            # nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
        )
        self.interaction_fake_fc1 = nn.Linear(hidden_dim*3, hidden_dim)
        self.interaction_fake_score = nn.Linear(hidden_dim, 222)

    def detectron_weight_mapping(self):
        # hc is human centric branch
        # io is interaction branch object part
        detectron_weight_mapping = {
            'human_fc1.weight': 'hc_fc1_w',
            'human_fc1.bias': 'hc_fc1_b',
            'human_fc2.weight': 'hc_fc2_w',
            'human_fc2.bias': 'hc_fc2_b',
            'human_action_score.weight': 'hc_score_w',
            'human_action_score.bias': 'hc_score_b',
            'interaction_fc1.weight': 'inter_fc1_w',
            'interaction_fc1.bias': 'inter_fc1_b',
        }
        return detectron_weight_mapping, []

    def fabricate_objs(self, verbs, labels, device_id):
        # import ipdb
        # ipdb.set_trace()
        obj_labels = torch.arange(0, OBJ_NUM).cuda(device_id)
        onehot_obj_labels = torch.zeros(OBJ_NUM, OBJ_NUM).cuda(device_id).scatter_(1, obj_labels.view(OBJ_NUM, 1), 1)
        # torch.ons
        obj_labels = torch.unsqueeze(obj_labels, dim=0)

        obj_labels = obj_labels.repeat([len(labels), 1])
        onehot_obj_labels = torch.unsqueeze(onehot_obj_labels, dim=0).repeat([len(labels), 1, 1])
        onehot_obj_labels = onehot_obj_labels.view([-1, OBJ_NUM])

        verbs = torch.unsqueeze(verbs, dim=1).repeat([1, OBJ_NUM, 1])
        verbs = torch.reshape(verbs, [len(verbs) * OBJ_NUM, -1])

        obj_labels = obj_labels.view(-1)
        obj_feats = self.obj_embedding(obj_labels)  # (b*nr_b*nr_f, coord_feature_dim//2)

        labels = torch.unsqueeze(labels, dim=1).repeat([1, OBJ_NUM, 1])
        labels = torch.reshape(labels, [len(labels) * OBJ_NUM, -1])
        noise = torch.randn([len(labels), 256]).cuda(device_id)
        import os
        faked_obj = torch.cat([verbs, obj_feats, noise], dim=-1)


        verb_obj_matrix = np.matmul(self.verb_to_HO_matrix, self.obj_to_HO_matrix.transpose())
        verb_obj_matrix = torch.from_numpy(verb_obj_matrix).float().cuda(device_id)
        # dim of verb

        ll = torch.matmul(labels, verb_obj_matrix)
        conds = torch.mul(onehot_obj_labels, ll)
        conds = torch.sum(conds, dim=-1) > 0

        orig_faked_obj = faked_obj
        orig_obj_labels = obj_labels
        faked_obj = faked_obj[conds]
        faked_obj = torch.cat([orig_faked_obj[:1], faked_obj], dim=0)  # avoid none list
        import os

        faked_obj = self.obj_fabricator(faked_obj)
        return faked_obj[1:], onehot_obj_labels[conds]

    def remove_vars(self, labels, vars, device_id):
        # import ipdb
        # ipdb.set_trace()
        for i in range(len(vars)):
            if len(vars[i].shape) == 1:
                vars[i] = torch.unsqueeze(vars[i], dim=1).repeat([1, OBJ_NUM])
                # vars[i] = torch.repeat_interleave(torch.unsqueeze(vars[i], dim=1), OBJ_NUM)
                vars[i] = torch.reshape(vars[i], [len(vars[i]) * OBJ_NUM])
            else:
                vars[i] = torch.unsqueeze(vars[i], dim=1).repeat([1, OBJ_NUM, 1])
                # vars[i] = torch.repeat_interleave(torch.unsqueeze(vars[i], dim=1), OBJ_NUM)
                vars[i] = torch.reshape(vars[i], [len(vars[i]) * OBJ_NUM, -1])

        old_labels = labels
        labels = torch.unsqueeze(labels, dim=1).repeat([1, OBJ_NUM, 1])
        labels = torch.reshape(labels, [len(labels) * OBJ_NUM, -1])

        obj_labels = torch.arange(0, OBJ_NUM).cuda(device_id)
        onehot_obj_labels = torch.zeros(OBJ_NUM, OBJ_NUM).cuda(device_id).scatter_(1, obj_labels.view(OBJ_NUM, 1), 1)
        # torch.ons

        onehot_obj_labels = torch.unsqueeze(onehot_obj_labels, dim=0).repeat([len(old_labels), 1, 1])
        onehot_obj_labels = onehot_obj_labels.view([-1, OBJ_NUM])

        verb_obj_matrix = np.matmul(self.verb_to_HO_matrix, self.obj_to_HO_matrix.transpose())
        verb_obj_matrix = torch.from_numpy(verb_obj_matrix).float().cuda(device_id)
        # dim of verb

        ll = torch.matmul(labels, verb_obj_matrix)
        conds = torch.mul(onehot_obj_labels, ll)
        conds = torch.sum(conds, dim=-1) > 0

        labels = labels[conds]
        for i in range(len(vars)):
            vars[i] = vars[i][conds]

        return vars + [labels]

    def forward(self, x, hoi_blob,):

        device_id = x[0].get_device()
        coord_x, coord_y = np.meshgrid(np.arange(x.shape[-1]), np.arange(x.shape[-2]))
        coords = np.stack((coord_x, coord_y), axis=0).astype(np.float32)
        coords = torch.from_numpy(coords).cuda(device_id) 
        x_coords = coords.unsqueeze(0).repeat(x.shape[0], 1, 1, 1) # 1 x 2 x H x W

        x_human = self.roi_xform(
            x, hoi_blob,
            blob_rois='human_boxes',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )

        x_object = self.roi_xform(
            x, hoi_blob,
            blob_rois='object_boxes',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )

        x_union = self.roi_xform(
            x, hoi_blob,
            blob_rois='union_boxes',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        x_union = x_union.view(x_union.size(0), -1)
        x_human = x_human.view(x_human.size(0), -1)
        x_object = x_object.view(x_object.size(0), -1)

        #x_object2 = x_object2.view(x_object2.size(0), -1)
        # get inds from numpy
        interaction_human_inds = torch.from_numpy(
            hoi_blob['interaction_human_inds']).long().cuda(device_id)
        interaction_object_inds = torch.from_numpy(
            hoi_blob['interaction_object_inds']).long().cuda(device_id)
        part_boxes = torch.from_numpy(
            hoi_blob['part_boxes']).cuda(device_id)

        x_human = F.relu(self.human_fc1(x_human[interaction_human_inds]), inplace=True)
        x_object = F.relu(self.object_fc1(x_object[interaction_object_inds]), inplace=True)

        # fabricating objects
        import os
        x_union = F.relu(self.union_fc1(x_union), inplace=True)
        if 'FABRICATOR' in os.environ and os.environ['FABRICATOR'].startswith('fcl'):
            interaction_action_labels = torch.from_numpy(hoi_blob['interaction_action_labels']).float().cuda(device_id)

            faked_obj, faked_obj_labels = self.fabricate_objs(x_human, interaction_action_labels, device_id)
            faked_x_human, faked_x_union, faked_verb, _ = self.remove_vars(interaction_action_labels, [x_human, x_union, interaction_action_labels], device_id)
            x_human = torch.cat([x_human, faked_x_human], dim=0)
            x_union = torch.cat([x_union, faked_x_union], dim=0)
            x_object = torch.cat([x_object, faked_obj], dim=0)

        x_interaction = torch.cat((x_human, x_object, x_union), dim=1)

        ## encode the pose information into x_interaction feature
        kps_pred = hoi_blob['poseconfig']
        if isinstance(kps_pred, np.ndarray):
            kps_pred = torch.from_numpy(kps_pred).cuda(device_id)
        poseconfig = kps_pred.view(kps_pred.size(0), -1)
        # x_pose_line = kps_pred.view(kps_pred.size(0), -1)
        x_pose_line = F.relu(self.pose_fc3(poseconfig), inplace=True)
        x_pose_line = F.relu(self.pose_fc4(x_pose_line), inplace=True)
        # x_interaction1 = torch.cat((x_interaction, x_pose_line), dim=1)  ## to get global interaction affinity score

        x_new = torch.cat((x, x_coords), dim=1)
        # x_new = x
        x_object2 = self.roi_xform(
            x_new, hoi_blob,
            blob_rois='object_boxes',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=self.crop_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )

        ## pose_attention feature, including part feature and object feature and geometry feature
        x_pose = self.crop_pose_map(x_new, part_boxes, hoi_blob['flag'], self.crop_size)
        # x_pose = torch.cat((x_pose, x_coords), dim=2) # N x 17 x 258 x 5 x 5
        x_pose = x_pose[interaction_human_inds]

        # x_object2 = torch.cat((x_object2, x_object2_coord), dim=1) # N x 258 x 5 x 5
        x_object2 = x_object2.unsqueeze(dim=1) # N x 1 x 258 x 5 x 5
        # N x 2 x 5 x 5

        x_object2 = x_object2[interaction_object_inds]
        center_xy = x_object2[:,:, -2:, 2:3, 2:3] # N x 1 x 2 x 1 x 1
        x_pose[:, :, -2:] = x_pose[:, :, -2:] - center_xy # N x 1 x 2 x 5 x 5
        x_object2[:,:,-2:] = x_object2[:,:, -2:] - center_xy # N x 17 x 2 x 5 x 5
        x_pose = torch.cat((x_pose, x_object2), dim=1) # N x 18 x 258 x 5 x 5
        # N x 18 x 256 x 5 x 5

        semantic_atten = F.sigmoid(self.mlp(poseconfig))
        semantic_atten = semantic_atten.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # N x 17 x 1 x 1 x 1
        x_pose_new = torch.zeros(x_pose.shape).cuda(device_id)
        x_pose_new[:, :17] = x_pose[:, :17] * semantic_atten
        x_pose_new[:, 17] = x_pose[:, 17]

        ## fuse the pose attention information
        x_pose = x_pose_new.view(x_pose_new.shape[0], -1)
        x_pose = F.relu(self.pose_fc1(x_pose), inplace=True)
        x_pose = F.relu(self.pose_fc2(x_pose), inplace=True)
        # expand feats
        # import ipdb
        # ipdb.set_trace()
        real_length = len(x_pose)
        if 'FABRICATOR' in os.environ and os.environ['FABRICATOR'].startswith('fcl'):
            interaction_action_labels = torch.from_numpy(hoi_blob['interaction_action_labels']).float().cuda(device_id)
            interaction_affinity = torch.from_numpy(hoi_blob['interaction_affinity']).float().cuda(device_id)
            faked_x_pose, faked_x_pose_line, interaction_affinity, interaction_action_labels = self.remove_vars(interaction_action_labels,
                                                                                                                [x_pose, x_pose_line, interaction_affinity], device_id)
            hoi_blob['interaction_orig_length'] = len(hoi_blob['interaction_affinity'])
            # import ipdb
            # ipdb.set_trace()
            if len(interaction_affinity.cpu().numpy()) > 0:
                x_pose = torch.cat([x_pose, faked_x_pose], dim=0)
                x_pose_line = torch.cat([x_pose_line, faked_x_pose_line])
                hoi_blob['interaction_affinity'] = np.concatenate([hoi_blob['interaction_affinity'], interaction_affinity.cpu().numpy()], axis=0)
                hoi_blob['interaction_action_labels'] = np.concatenate([hoi_blob['interaction_action_labels'], interaction_action_labels.cpu().numpy()], axis=0)


        x_interaction = torch.cat((x_interaction, x_pose, x_pose_line), dim=1)
        if 'FABRICATOR' in os.environ and os.environ['FABRICATOR'].startswith('fcl'):
            real_x_interaction = x_interaction[:real_length]
            fake_x_interaction = x_interaction[real_length:]
            # randomly sample len(real_x_interaction) HOIs
            select_indx = torch.randperm(len(fake_x_interaction))
            rand_length = len(real_x_interaction)

            select_indx = select_indx[:rand_length]
            select_indx = select_indx.cuda(device_id)
            fake_x_interaction = fake_x_interaction[select_indx]
            x_interaction = torch.cat([real_x_interaction, fake_x_interaction], dim=0)

            hoi_blob['interaction_affinity'] = np.concatenate(
                [hoi_blob['interaction_affinity'][:real_length], hoi_blob['interaction_affinity'][real_length:][select_indx.cpu().numpy()]], axis=0)
            hoi_blob['interaction_action_labels'] = np.concatenate(
                [hoi_blob['interaction_action_labels'][:real_length], hoi_blob['interaction_action_labels'][real_length:][select_indx.cpu().numpy()]], axis=0)

        interaction_affinity_score = self.global_affinity(x_interaction)

        x_interaction = F.relu(self.interaction_fc1(x_interaction), inplace=True)
        interaction_action_score = self.interaction_action_score(x_interaction)

        hoi_blob['interaction_action_score'] = interaction_action_score  ### multi classification score
        hoi_blob['interaction_affinity_score']= interaction_affinity_score ### binary classisification score

        return hoi_blob

    def crop_pose_map(self, union_feats, part_boxes, flag, crop_size):
        triplets_num, part_num, _ = part_boxes.shape
        ret = torch.zeros((triplets_num, part_num, union_feats.shape[1], crop_size, crop_size)).cuda(
            union_feats.get_device())
        part_feats = RoIAlignFunction(crop_size, crop_size, self.spatial_scale, cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO)(
            union_feats, part_boxes.view(-1, part_boxes.shape[-1])).view(ret.shape)

        valid_n, valid_p = np.where(flag > 0)
        if len(valid_n) > 0:
            ret[valid_n, valid_p] = part_feats[valid_n, valid_p]
        return ret

    @staticmethod
    def loss(hoi_blob):

        interaction_action_score = hoi_blob['interaction_action_score']
        interaction_affinity_score = hoi_blob['interaction_affinity_score']
        device_id = interaction_action_score.get_device()

        interaction_action_labels = torch.from_numpy(hoi_blob['interaction_action_labels']).float().cuda(device_id)
        interaction_action_preds = \
            (interaction_action_score.sigmoid() > cfg.VCOCO.ACTION_THRESH).type_as(interaction_action_labels)

        # import ipdb
        # ipdb.set_trace()
        import os
        if 'FABRICATOR' in os.environ and os.environ['FABRICATOR'].startswith('fcl'):
            interaction_action_loss = F.binary_cross_entropy_with_logits(
                interaction_action_score, interaction_action_labels, reduction='none')
            orig_length = hoi_blob['interaction_orig_length']
            l = 0.2 # or 0.25: this is following the setting on HICO-DET( 0.5 / 2)
            interaction_action_loss = (torch.sum(interaction_action_loss[:orig_length]) + torch.sum(interaction_action_loss[orig_length:]) * l) / (orig_length*24 + len(interaction_action_loss[orig_length:])*l*24)

        else:
            interaction_action_loss = F.binary_cross_entropy_with_logits(
                interaction_action_score, interaction_action_labels, size_average=True)

        interaction_action_accuray_cls = interaction_action_preds.eq(interaction_action_labels).float().mean()
        interaction_affinity_label = torch.from_numpy(hoi_blob['interaction_affinity'].astype(np.float32)).cuda(
            device_id)
        # interaction_affinity_loss = F.cross_entropy(
        #     interaction_affinity_score, interaction_affinity_label)
        # interaction_affinity_preds = (interaction_affinity[:,1]>interaction_affinity[:,0]).type_as(interaction_affinity_label)
        l = 1. # AFFINITY_WEIGHT is 0.1, thus we simply set l as 1.
        # interaction_affinity_loss = cfg.VCOCO.AFFINITY_WEIGHT * (torch.sum(interaction_affinity_loss[:orig_length]) + torch.sum(interaction_affinity_loss[orig_length:]) * l) / (orig_length + len(interaction_affinity_loss[orig_length:])*l)
        interaction_affinity_loss = cfg.VCOCO.AFFINITY_WEIGHT * F.binary_cross_entropy_with_logits(
            interaction_affinity_score, interaction_affinity_label.unsqueeze(1), size_average=True)
        interaction_affinity_preds = (interaction_affinity_score.sigmoid() > cfg.VCOCO.ACTION_THRESH).type_as(
            interaction_affinity_label)
        interaction_affinity_cls = interaction_affinity_preds.eq(interaction_affinity_label).float().mean()

        return interaction_action_loss, interaction_affinity_loss, \
               interaction_action_accuray_cls, interaction_affinity_cls


class PMFNet_Final_bak(nn.Module):
    """
    add relative coordinate to parts
    Human Object Interaction.
    This module including Human-centric branch and Interaction branch of InteractNet.
    """

    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()

        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.crop_size = cfg.VCOCO.PART_CROP_SIZE

        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        # ToDo: cfg
        hidden_dim = cfg.VCOCO.MLP_HEAD_DIM
        # num_action_classes = cfg.VCOCO.NUM_ACTION_CLASSES
        action_mask = np.array(cfg.VCOCO.ACTION_MASK).T
        interaction_num_action_classes = action_mask.sum().item()

        interaction_fc1_dim_in = 3 * hidden_dim
        part_num = 17
        self.pose_fc1 = nn.Linear((part_num + 1) * 258 * self.crop_size ** 2, 1024)
        self.pose_fc2 = nn.Linear(1024, hidden_dim * 2)
        interaction_fc1_dim_in += hidden_dim * 2

        self.mlp = nn.Sequential(
            nn.Linear(3 * 64 * 64, 64),
            nn.ReLU(),
            nn.Linear(64, part_num)
        )

        self.pose_fc3 = nn.Linear(3 * cfg.KRCNN.HEATMAP_SIZE ** 2, 512)
        self.pose_fc4 = nn.Linear(512, hidden_dim)
        interaction_fc1_dim_in += hidden_dim

        self.human_fc1 = nn.Linear(dim_in * roi_size ** 2, hidden_dim)
        self.object_fc1 = nn.Linear(dim_in * roi_size ** 2, hidden_dim)
        self.union_fc1 = nn.Linear(dim_in * roi_size ** 2, hidden_dim)

        self.interaction_fc1 = nn.Linear(interaction_fc1_dim_in, hidden_dim)
        self.interaction_action_score = nn.Linear(hidden_dim, interaction_num_action_classes)

    def detectron_weight_mapping(self):
        # hc is human centric branch
        # io is interaction branch object part
        detectron_weight_mapping = {
            'human_fc1.weight': 'hc_fc1_w',
            'human_fc1.bias': 'hc_fc1_b',
            'human_fc2.weight': 'hc_fc2_w',
            'human_fc2.bias': 'hc_fc2_b',
            'human_action_score.weight': 'hc_score_w',
            'human_action_score.bias': 'hc_score_b',
            'interaction_fc1.weight': 'inter_fc1_w',
            'interaction_fc1.bias': 'inter_fc1_b',
        }
        return detectron_weight_mapping, []

    def forward(self, x, hoi_blob):

        device_id = x[0].get_device()

        coord_x, coord_y = np.meshgrid(np.arange(x.shape[-1]), np.arange(x.shape[-2]))
        coords = np.stack((coord_x, coord_y), axis=0).astype(np.float32)
        coords = torch.from_numpy(coords).cuda(device_id)
        x_coords = coords.unsqueeze(0)  # 1 x 2 x H x W

        x_human = self.roi_xform(
            x, hoi_blob,
            blob_rois='human_boxes',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )

        x_object = self.roi_xform(
            x, hoi_blob,
            blob_rois='object_boxes',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )

        x_new = torch.cat((x, x_coords), dim=1)
        x_object2 = self.roi_xform(
            x_new, hoi_blob,
            blob_rois='object_boxes',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=self.crop_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )

        x_human = x_human.view(x_human.size(0), -1)
        x_object = x_object.view(x_object.size(0), -1)

        # x_object2 = x_object2.view(x_object2.size(0), -1)
        # get inds from numpy
        interaction_human_inds = torch.from_numpy(
            hoi_blob['interaction_human_inds']).long().cuda(device_id)
        interaction_object_inds = torch.from_numpy(
            hoi_blob['interaction_object_inds']).long().cuda(device_id)
        part_boxes = torch.from_numpy(
            hoi_blob['part_boxes']).cuda(device_id)

        x_human = F.relu(self.human_fc1(x_human[interaction_human_inds]), inplace=True)
        x_object = F.relu(self.object_fc1(x_object[interaction_object_inds]), inplace=True)

        x_union = self.roi_xform(
            x, hoi_blob,
            blob_rois='union_boxes',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )

        x_union = x_union.view(x_union.size(0), -1)
        x_union = F.relu(self.union_fc1(x_union), inplace=True)
        x_interaction = torch.cat((x_human, x_object, x_union), dim=1)



        kps_pred = hoi_blob['poseconfig']
        if isinstance(kps_pred, np.ndarray):
            kps_pred = torch.from_numpy(kps_pred).cuda(device_id)

        poseconfig = kps_pred.view(kps_pred.size(0), -1)
        x_pose = self.crop_pose_map(x_new, part_boxes, hoi_blob['flag'], self.crop_size)

        # x_pose = torch.cat((x_pose, x_coords), dim=2) # N x 17 x 258 x 5 x 5
        x_pose = x_pose[interaction_human_inds]

        # x_object2 = torch.cat((x_object2, x_object2_coord), dim=1) # N x 258 x 5 x 5
        x_object2 = x_object2.unsqueeze(dim=1)  # N x 1 x 258 x 5 x 5
        # N x 2 x 5 x 5

        x_object2 = x_object2[interaction_object_inds]
        center_xy = x_object2[:, :, -2:, 2:3, 2:3]  # N x 1 x 2 x 1 x 1
        x_pose[:, :, -2:] = x_pose[:, :, -2:] - center_xy  # N x 1 x 2 x 5 x 5
        x_object2[:, :, -2:] = x_object2[:, :, -2:] - center_xy  # N x 17 x 2 x 5 x 5

        x_pose = torch.cat((x_pose, x_object2), dim=1)  # N x 18 x 258 x 5 x 5

        semantic_atten = F.sigmoid(self.mlp(poseconfig))
        semantic_atten = semantic_atten.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # N x 17 x 1 x 1 x 1
        x_pose_new = torch.zeros(x_pose.shape).cuda(device_id)
        x_pose_new[:, :17] = x_pose[:, :17] * semantic_atten
        x_pose_new[:, 17] = x_pose[:, 17]

        x_pose = x_pose_new.view(x_pose_new.shape[0], -1)
        x_pose = F.relu(self.pose_fc1(x_pose), inplace=True)
        x_pose = F.relu(self.pose_fc2(x_pose), inplace=True)
        x_interaction = torch.cat((x_interaction, x_pose), dim=1)

        x_pose_line = kps_pred.view(kps_pred.size(0), -1)
        x_pose_line = F.relu(self.pose_fc3(x_pose_line), inplace=True)
        x_pose_line = F.relu(self.pose_fc4(x_pose_line), inplace=True)
        x_interaction = torch.cat((x_interaction, x_pose_line), dim=1)

        x_interaction = F.relu(self.interaction_fc1(x_interaction), inplace=True)
        interaction_action_score = self.interaction_action_score(x_interaction)

        hoi_blob['interaction_action_score'] = interaction_action_score
        hoi_blob['interaction_affinity_score'] = torch.zeros((x_union.shape[0], 2)).cuda(device_id)

        return hoi_blob

    def crop_pose_map(self, union_feats, part_boxes, flag, crop_size):
        triplets_num, part_num, _ = part_boxes.shape
        ret = torch.zeros((triplets_num, part_num, union_feats.shape[1], crop_size, crop_size)).cuda(
            union_feats.get_device())
        part_feats = RoIAlignFunction(crop_size, crop_size, self.spatial_scale, cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO)(
            union_feats, part_boxes.view(-1, part_boxes.shape[-1])).view(ret.shape)

        valid_n, valid_p = np.where(flag > 0)
        if len(valid_n) > 0:
            ret[valid_n, valid_p] = part_feats[valid_n, valid_p]
        # return ret.reshape(triplets_num, part_num, -1)
        return ret

    @staticmethod
    def loss(hoi_blob):

        interaction_action_score = hoi_blob['interaction_action_score']
        device_id = interaction_action_score.get_device()

        ''' for fine_grained action loss'''
        interaction_action_labels = torch.from_numpy(hoi_blob['interaction_action_labels']).float().cuda(device_id)
        interaction_action_loss = F.binary_cross_entropy_with_logits(
            interaction_action_score, interaction_action_labels)
        # get interaction branch predict action accuracy
        interaction_action_preds = \
            (interaction_action_score.sigmoid() > cfg.VCOCO.ACTION_THRESH).type_as(interaction_action_labels)
        interaction_action_accuray_cls = interaction_action_preds.eq(interaction_action_labels).float().mean()

        return interaction_action_loss, interaction_action_accuray_cls