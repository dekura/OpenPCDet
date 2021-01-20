python=/home/guojin/miniconda3/envs/pcdet/bin/python

model=hsd_pv_rcnn

$python  demo.py --cfg_file cfgs/kitti_models/$model.yaml \
    --ckpt /home/guojin/projects/pchsd/OpenPCDet/output/cfgs/kitti_models/$model/default/ckpt/checkpoint_epoch_200.pth \
    --data_path /home/guojin/projects/pchsd/OpenPCDet/data/hsd_kitti/training/velodyne/000019.bin \
    --output /home/guojin/projects/pchsd/OpenPCDet/demo_opt

# $python  demo.py --cfg_file cfgs/kitti_models/hsd_pv_rcnn.yaml \
#     --ckpt /home/guojin/projects/pchsd/OpenPCDet/output/cfgs/kitti_models/hsd_pv_rcnn/default/ckpt/checkpoint_epoch_200.pth \
#     --data_path /home/guojin/projects/pchsd/OpenPCDet/data/hsd_kitti/training/velodyne/000019.bin \
#     --output /home/guojin/projects/pchsd/OpenPCDet/demo_opt


scp -r /home/guojin/projects/pchsd/OpenPCDet/demo_opt/ dekuraMac:/Users/dekura/chen/bei/projects/pchsd/OpenPCDet/
echo 'scp to dekura@Mac done'