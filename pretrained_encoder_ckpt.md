## Checkpoints

可以将我们的对比学习（stage 1）分为两种方式：WirelessBind和ImageBind，前者以Beam为锚点，后者以视觉为锚点。

具体到训练过程：
```python
if anchor_modality == 'vision':
    loss_vision_mmwave = info_nce_loss(vision_features, mmwave_features)
    loss_vision_gps = info_nce_loss(vision_features, gps_features)
    loss = loss_vision_mmwave + loss_vision_gps
elif anchor_modality == 'mmwave':
    loss_mmwave_vision = info_nce_loss(mmwave_features, vision_features)
    loss_mmwave_gps = info_nce_loss(mmwave_features, gps_features)
    loss = loss_mmwave_vision + loss_mmwave_gps
```

### Checkpoint 文件命名:
- mmwave_gps_xxx: WirelessBind
- vision_xxx: ImageBind
- s2/s4/s6: sn表示使用数据集中的前n个场景训练