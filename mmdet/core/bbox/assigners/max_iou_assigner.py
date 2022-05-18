# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class MaxIoUAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
            `min_pos_iou` is set to avoid assigning bboxes that have extremely
            small iou with GT as positive samples. It brings about 0.3 mAP
            improvements in 1x schedule but does not affect the performance of
            3x schedule. More comparisons can be found in
            `PR #7464 <https://github.com/open-mmlab/mmdetection/pull/7464>`_.
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
        match_low_quality (bool): Whether to allow low quality matches. This is
            usually allowed for RPN and single stage detectors, but not allowed
            in the second stage. Details are demonstrated in Step 4.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
        1、计算所有bbox与所有gt的iou。
        2、计算所有bbox与所有gt_ignore的iof，若某个bbox与所有gt_ignore的iof大于ignore_iof_thr ，
        则把该bbox与所有gt的iou都设为-1。
        3、初始化，把所有bbox设置为ignore忽略样本，assigned_gt_inds赋值为-1。
        4、获得每个bbox的最近的gt（即argmax gt），若某个bbox与最近的gt的iou（即max iou）都小于neg_iou_thr，
        那么设置为负样本，assigned_gt_inds赋值为0。
        5、若某个bbox与最近的gt的iou（即max iou）大于pos_iou_thr，那么设置为正样本，assigned_gt_inds赋值为
        该bbox最近的gt（argmax gt）的下标+1，（范围为1~len(gts)）。
        6、（可选步骤，match_low_quality为True），为什么要这步操作呢？因为可能存在某些gt没有被正样本匹配到，
        这时我们放宽限制。先对每个gt获得最近的bbox，若某个gt与最近bbox的iou大于min_pos_iou，
        那么该最近bbox（即argmax bbox）设置为gt的下标+1。若与该gt最近的bbox有多个时，
        我们可以根据gt_max_assign_all确定是否对所有最近bbox还是某个最近bbox赋值。
    """

    def __init__(self,
                 pos_iou_thr,  # max iou大于该值为正样本
                 neg_iou_thr,  # max iou 小于该值为负样本
                 min_pos_iou=.0,  # match_low_quality 为True时，max iou大于改值时，为正样本
                 gt_max_assign_all=True,  # match_low_quality为True时，是否对gt所有argmax bbox进行赋值
                 ignore_iof_thr=-1,  # 若bbox与gt_ignore的iof大于该值，则该bbox设置为忽略样本
                 ignore_wrt_candidates=True,  # 确定iof的前景
                 match_low_quality=True,  # 是否启用第四步
                 gpu_assign_thr=-1,  # 节省显存的，若gt数量大于gpu_assign_thr,则放到cpu上，-1的话就是全部放到gpu
                 iou_calculator=dict(type='BboxOverlaps2D')):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.match_low_quality = match_low_quality
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, or a semi-positive number. -1 means negative
        sample, semi-positive number is the index (0-based) of assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to the background
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.

        Example:
            >>> self = MaxIoUAssigner(0.5, 0.5)
            >>> bboxes = torch.Tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
            >>> gt_bboxes = torch.Tensor([[0, 0, 10, 9]])
            >>> assign_result = self.assign(bboxes, gt_bboxes)
            >>> expected_gt_inds = torch.LongTensor([1, 0])
            >>> assert torch.all(assign_result.gt_inds == expected_gt_inds)
        """
        # 主要是为了节省显存，iou计算还是很耗显存的，上万个bbox呢
        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
                gt_bboxes.shape[0] > self.gpu_assign_thr) else False
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = bboxes.device
            bboxes = bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()

        # 第一步 计算所有bbox与所有gt的IOU
        overlaps = self.iou_calculator(gt_bboxes, bboxes)
        # 第二部 计算bbox与gt_ignore的iof
        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            if self.ignore_wrt_candidates:  # 为True，则为前景为bboxes，否则前景为gt_bboxes
                ignore_overlaps = self.iou_calculator(
                    bboxes, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(
                    gt_bboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1
            # 直接把与ignore区域大于阈值的都设置为-1

        # 为每一个bboxes分配gt_bboxes和对应的类
        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)

        # 映射回Gpu设备
        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result

    def assign_wrt_overlaps(self, overlaps, gt_labels=None):
        """Assign w.r.t. the overlaps of bboxes with gts.

        Args:
            overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
                shape(k, n).
            gt_labels (Tensor, optional): Labels of k gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 初始化全为-1，表示忽略样本
        # 1. assign -1 by default
        # 表示每一个anchor对应那种样本，-1表示忽略，0表示负样本，1~gt表示是对应gt的正样本
        assigned_gt_inds = overlaps.new_full((num_bboxes,),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes,),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        # 每一个anchor对应的gt_boxes的iou值以及gt下标
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        # 2. assign negative: below
        # the negative inds are set to be 0
        # 分配负样本
        if isinstance(self.neg_iou_thr, float):
            # 不是忽略样本且iou小于neg阈值
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                             & (max_overlaps < self.neg_iou_thr[1])] = 0

        # 3. assign positive: above positive IoU threshold
        # 大于正样本阈值是正样本，并且将gt下标+1赋值给assigned_gt_inds
        pos_inds = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        # 虽然已经为每一个anchor都有对应的划分，但是并不是每一个gt都有之对应的anchor
        if self.match_low_quality:
            # Low-quality matching will overwrite the assigned_gt_inds assigned
            # in Step 3. Thus, the assigned gt might not be the best one for
            # prediction.
            # For example, if bbox A has 0.9 and 0.8 iou with GT bbox 1 & 2,
            # bbox 1 will be assigned as the best target for bbox A in step 3.
            # However, if GT bbox 2's gt_argmax_overlaps = A, bbox A's
            # assigned_gt_inds will be overwritten to be bbox 2.
            # This might be the reason that it is not used in ROI Heads.
            for i in range(num_gts):
                # 每一个gt最大iou如果大于该阈值
                if gt_max_overlaps[i] >= self.min_pos_iou:
                    # 是否大于该阈值的全要
                    if self.gt_max_assign_all:
                        max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                        assigned_gt_inds[max_iou_inds] = i + 1
                    else:
                        assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        if gt_labels is not None:
            # assigned_labels 表示正样本对应的类别，范围为[-1~len(class)-1]
            # -1表示非正样本(包括负样本和忽略样本)
            # 0~len(class)-1就是正样本对应的class
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            # 取出正样本的所有下标
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                # 为正样本赋值对应的类别
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        # 封装成这个类
        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
