# model settings


model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ESTF',
        pretrained=None,
        batchNorm=True, 
        output_layers=None,
        init_std=0.005,

        dim = 256, 
        clip_len=8,
        num_heads = 4,
        mlp_ratio=4., 
        qkv_bias=False, 
        drop=0., 
        attn_drop=0., 
        init_values=1e-5,
        drop_path=0., 
        to_device="cuda:0",
        ),

    cls_head=dict(
        type='I3DHead',
        # num_classes=1000,
        # num_classes=300,
        num_classes=101,
        # in_channels=4096,
        in_channels=3072,
        spatial_type=None,
        dropout_ratio=0.5,
        init_std=0.01),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='score'))