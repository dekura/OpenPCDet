python=/home/guojin/miniconda3/envs/pchsd/bin/python

$python  demo.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml \
    --ckpt /home/guojin/projects/pchsd/OpenPCDet/checkpoints/pvrcnn/pv_rcnn_8369.pth \
    --data_path /home/guojin/data/datasets/kitti/training/velodyne/004987.bin \
    --output /home/guojin/projects/pchsd/OpenPCDet/outputs
