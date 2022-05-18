# model settings
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        # 表示本模块输出的特征图索引，(0, 1, 2, 3),表示4个 stage 输出都需要，
        # 其 stride 为 (4,8,16,32)，channel 为 (256, 512, 1024, 2048)
        # 例如如果你想要输出 stride=4 的特征图，那么你可以设置 out_indices=(0,)，
        # 如果你想要输出 stride=4 和 8 的特征图，那么你可以设置 out_indices=(0, 1)
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=20,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            # 特征图 anchor 的 base scale, 值越大，所有 anchor 的尺度都会变大
            octave_base_scale=4,
            # 每个特征图有3个尺度，2**0, 2**(1/3), 2**(2/3)
            scales_per_octave=3,
            # 每个特征图有3个高宽比例
            ratios=[0.5, 1.0, 2.0],
            # 特征图对应的 stride，必须特征图 stride 一致，不可以随意更改
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            # 正样本阈值
            pos_iou_thr=0.5,
            # 负样本阈值
            neg_iou_thr=0.4,
            # 正样本阈值下限
            min_pos_iou=0,
            # 忽略bboxes的阈值，-1表示不忽略
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
