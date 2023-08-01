auto_scale_lr = dict(base_batch_size=16)
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    num_classes=158,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
dataset_type = 'Garbage'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'))
default_scope = 'mmcls'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
log_level = 'INFO'
load_from = '../efficientnet-b1_3rdparty-ra-noisystudent_in1k_20221103-756bcbc0.pth'
model = dict(
    backbone=dict(arch='b1', type='EfficientNet'),
    head=dict(
        in_channels=1280,
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss'),
        num_classes=158,
        topk=(
            1,
            5,
        ),
        type='LinearClsHead'),
    neck=dict(type='GlobalAveragePooling'),
    type='ImageClassifier')
optim_wrapper = dict(
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0001))
param_scheduler = dict(
    by_epoch=True, gamma=0.1, milestones=[
        2,
        5,
        8,
    ], type='MultiStepLR')
randomness = dict(deterministic=False, seed=None)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=16,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='test.txt',
        data_root='../../garbage',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackClsInputs'),
        ],
        split='',
        type='Garbage'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(topk=(1, ), type='Accuracy')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(crop_size=224, type='CenterCrop'),
    dict(type='PackClsInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=10, val_interval=1)
train_dataloader = dict(
    batch_size=16,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='train.txt',
        data_root='../../garbage',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=224, type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackClsInputs'),
        ],
        split='',
        type='Garbage'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(scale=224, type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(type='PackClsInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=16,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='val.txt',
        data_root='../../garbage',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackClsInputs'),
        ],
        type='ImageNet'),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    topk=(
        1,
        5,
    ), type='Accuracy')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ClsVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '../work_dir/garbage'
