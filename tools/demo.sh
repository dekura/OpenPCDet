python=/home/guojin/miniconda3/envs/pchsd/bin/python

# $python  demo.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml \
#     --ckpt /home/guojin/projects/pchsd/OpenPCDet/checkpoints/pvrcnn/pv_rcnn_8369.pth \
#     --data_path /home/guojin/data/datasets/kitti/training/velodyne/004987.bin \
#     --output /home/guojin/projects/pchsd/OpenPCDet/outputs



# $python  demo.py --cfg_file cfgs/kitti_models/hsd_pointrcnn.yaml \
#     --ckpt /home/guojin/projects/pchsd/OpenPCDet/output/home/guojin/projects/pchsd/OpenPCDet/tools/cfgs/kitti_models/hsd_pointrcnn/default/ckpt/checkpoint_epoch_299.pth \
#     --data_path /home/guojin/projects/pchsd/OpenPCDet/data/hsd_kitti/training/velodyne/000000.bin \
#     --output /home/guojin/projects/pchsd/OpenPCDet/demo_opt



# $python  demo.py --cfg_file cfgs/kitti_models/hsd_second.yaml \
#     --ckpt /home/guojin/projects/pchsd/OpenPCDet/output/cfgs/kitti_models/hsd_second/default/ckpt/checkpoint_epoch_300.pth \
#     --data_path /home/guojin/projects/pchsd/OpenPCDet/data/hsd_kitti/training/velodyne/000000.bin \
#     --output /home/guojin/projects/pchsd/OpenPCDet/demo_opt


$python  demo.py --cfg_file cfgs/kitti_models/hsd_pointrcnn.yaml \
    --ckpt /home/guojin/projects/pchsd/OpenPCDet/output/cfgs/kitti_models/hsd_pointrcnn/default/ckpt/checkpoint_epoch_200.pth \
    --data_path /home/guojin/projects/pchsd/OpenPCDet/data/hsd_kitti/training/velodyne/000002.bin \
    --output /home/guojin/projects/pchsd/OpenPCDet/demo_opt


scp -r /home/guojin/projects/pchsd/OpenPCDet/demo_opt/ dekuraMac:/Users/dekura/chen/bei/projects/pchsd/OpenPCDet/
echo 'scp to dekura@Mac done'