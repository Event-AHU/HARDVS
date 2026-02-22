preprocess_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='vHeat',
        pretrained='torchvision://resnet50',
        is_shift=True,
        num_segments=8,
        depth=50,
        norm_eval=False,
        shift_div=8),
    cls_head=dict(
        type='TSMHead',
        num_classes=300,
        in_channels=768, # 2240
        num_segments=16,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        is_shift=True,
        average_clips='prob'),
    data_preprocessor=dict(type='ActionDataPreprocessor', **preprocess_cfg),
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))