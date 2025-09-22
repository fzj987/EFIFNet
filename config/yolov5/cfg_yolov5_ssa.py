default_scope = 'mmyolo'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='linear', #cosine   linear
        #warmup_epochs=3,
        #warmup_mim_iter=1000,
        lr_factor=0.01, #0.01
        max_epochs=400),#400
    checkpoint=dict(
        type='CheckpointHook', interval=5, save_best='coco/bbox_mAP',  # bbox_mAP   auto
        max_keep_ckpts=5),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='mmdet.DetVisualizationHook',
        draw=True,
        test_out_dir='./show_results'))
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
#load_from = '/home/qwe/zhh/EAEFNet/EAEFNet_Detection/EAEF_det_weight/bbox_mAP_epoch_300.pth'
load_from = '/home/qwe/zhh/EAEFNet/EAEFNet_Detection/EAEF_mmyolo/run/Eformer/yolov5_cs03_ssa2_ssa2_hdb1_sgd_400_2/best_coco_Car_precision_epoch_376.pth'
#load_from = '/home/qwe/zhh/EAEFNet/EAEFNet_Detection/EAEF_mmyolo/run/Eformer/yolov5_cs03_ssa1_ssa1_ssa1_sgd_finetune50/yolov5_finetune.pth'
#load_from = '/home/qwe/zhh/EAEFNet/EAEFNet_Detection/EAEF_mmyolo/ckpt/yolov5_m-v61_syncbn_fast_8xb16-300e_coco.pth'
resume = False
file_client_args = dict(backend='disk')
data_root = '/home/qwe/zhh/RGBT/M3FD'
dataset_type = 'YOLOv5CocoDataset'
num_classes = 6
img_scale = (640, 640)
deepen_factor = 0.67
widen_factor = 0.75
#deepen_factor = 0.33
#widen_factor = 0.5
max_epochs = 400
save_epoch_intervals = 5
train_batch_size_per_gpu = 8
train_num_workers = 2
val_batch_size_per_gpu = 2
val_num_workers = 2
persistent_workers = True
batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=1,
    img_size=640,
    size_divisor=32,
    extra_pad_ratio=0.5)
anchors = [[(10, 13), (16, 30), (33, 23)], 
           [(30, 61), (62, 45), (59, 119)],
           [(116, 90), (156, 198), (373, 326)]]
featmap_strides = [8, 16, 32]
num_det_layers = 3
model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[0.0,0.0,0.0,0.0],
        std=[255.0,255.0,255.0,255.0],
        bgr_to_rgb=True),
    backbone=dict(
        type='BiYOLOv5CSPDarknet',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        type='YOLOv5PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 1024],
        out_channels=[256, 512, 1024],#s1[48, 120, 224] l[80, 192, 384] s2[32, 64, 144, 288]
        num_csp_blocks=3,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='YOLOv5Head',
        head_module=dict(
            type='YOLOv5HeadModule',
            num_classes=6,
            in_channels=[256, 512, 1024],#s1[48, 120, 224] l[80, 192, 384] s2[32, 64, 144, 288]
            widen_factor=widen_factor,
            featmap_strides=featmap_strides,
            num_base_priors=3),
        prior_generator=dict(
            type='mmdet.YOLOAnchorGenerator',
            base_sizes=anchors,
            strides=featmap_strides),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=0.3),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xywh',
            eps=1e-07,
            reduction='mean',
            loss_weight=0.05,
            return_iou=True),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=0.7),
        prior_match_thr=4.0,
        obj_level_weights=[4.0, 1.0, 0.4]),
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root = '/home/qwe/zhh/RGBT/M3FD',
        data_prefix=dict(img='train/'),
        ann_file='instances_train2014.json',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=[
            dict(type='LoadImage'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Mosaic',
                img_scale=(640, 640),
                pad_val=114.0,
                pre_transform=[
                    dict(type='LoadImage'),
                    dict(type='LoadAnnotations', with_bbox=True)
                ]),
            dict(
                type='YOLOv5RandomAffine',
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                scaling_ratio_range=(0.5, 1.5),
                border=(-320, -320),
                border_val=(114, 114, 114)),
            dict(
                type='mmdet.Albu',
                transforms=[
                    dict(type='Blur', p=0.01),
                    dict(type='MedianBlur', p=0.01)
                ],
                bbox_params=dict(
                    type='BboxParams',
                    format='pascal_voc',
                    label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
                keymap=dict(img='image', gt_bboxes='bboxes')),

            dict(type='mmdet.RandomFlip', prob=0.5),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'flip', 'flip_direction'))
        ],
        metainfo=dict(
            classes=('Car', 'Truck', 'People', 'Bus', 'Lamp', 'Motorcycle'),
            palette=[(220, 20, 60), (255, 0, 255), (0, 255, 255), (0, 0, 255),
                     (183, 43, 42), (123, 23, 32)])),
    collate_fn=dict(type='yolov5_collate'))

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root = '/home/qwe/zhh/RGBT/M3FD',
        test_mode=True,
        data_prefix=dict(img='val/'),
        ann_file='instances_val2014.json',
        pipeline=[
            dict(type='LoadImage'),
            dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
            dict(
                type='LetterResize',
                scale=(640, 640),
                allow_scale_up=False,
                pad_val=dict(img=114)),
            dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor', 'pad_param'))
        ],
        batch_shapes_cfg=dict(
            type='BatchShapePolicy',
            batch_size=1,
            img_size=640,
            size_divisor=32,
            extra_pad_ratio=0.5),
        metainfo=dict(
            classes=('Car', 'Truck', 'People', 'Bus', 'Lamp', 'Motorcycle'),
            palette=[(220, 20, 60), (255, 0, 255), (0, 255, 255), (0, 0, 255),
                     (183, 43, 42), (123, 23, 32)])))

test_dataloader = val_dataloader


param_scheduler=None

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.01,#0.01
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True,
        batch_size_per_gpu=train_batch_size_per_gpu),
#    optimizer=dict(
#        type='AdamW',
#        lr=0.0002, weight_decay=0.05,
#        batch_size_per_gpu=1),
    constructor='YOLOv5OptimizerConstructor')

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49)
]
val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file='/home/qwe/zhh/RGBT/M3FD/instances_val2014.json',
    classwise=True,
    metric='bbox')

test_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file='/home/qwe/zhh/RGBT/M3FD/instances_val2014.json',
    classwise=True,
    metric='bbox')
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=save_epoch_intervals)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
metainfo=dict(
        classes=('Car', 'Truck', 'People', 'Bus', 'Lamp', 'Motorcycle'),
        palette=[(220, 20, 60), (255, 0, 255), (0, 255, 255), (0, 0, 255),
                     (183, 43, 42), (123, 23, 32)])

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend'),
                  dict(type='WandbVisBackend')],
    name='visualizer')

launcher = 'none'
work_dir = './work_dirs/bi_idam_yolov5'
