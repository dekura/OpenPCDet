# cuda error
python train.py --cfg_file ./cfgs/kitti_models/hsd_PartA2_free.yaml
# cuda error
python train.py --cfg_file ./cfgs/kitti_models/hsd_PartA2.yaml

# can train, results = 0
python train.py --cfg_file ./cfgs/kitti_models/hsd_pointpillar.yaml

# not try yet
python train.py --cfg_file ./cfgs/kitti_models/hsd_pointrcnn_iou.yaml

# cuda error
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/hsd_kitti_dataset.yaml

rm -rf output/cfgs/kitti_models/hsd_pointrcnn

python train.py --cfg_file ./cfgs/kitti_models/hsd_pointrcnn.yaml

# cuda error
python train.py --cfg_file ./cfgs/kitti_models/hsd_pv_rcnn.yaml

python train.py --cfg_file ./cfgs/kitti_models/hsd_second_multihead.yaml

# can train results = 0
python train.py --cfg_file ./cfgs/kitti_models/hsd_second.yaml
