import os
import glob
import torch
import pickle
import shutil
import argparse

import numpy as np

from tqdm import tqdm
from pathlib import Path

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

if os.name == 'posix' and "DISPLAY" not in os.environ:
    headless_server = True
else:
    headless_server = False
    from visual_utils import visualize_utils as V
    import mayavi.mlab as mlab

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--output', type=str, default=None, help='the output folder')


    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    export_dir = args.output
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info('Total number of samples: {}'.format(len(demo_dataset)))
    logger.info(args.data_path)

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info('Visualized sample index: {}'.format(idx+1))
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            if headless_server:
                # image_name = demo_dataset.sample_file_list[idx].split('/')[-1].replace(demo_dataset.ext, '.png')
                # image_source_path = image_path / image_name

                # if args.copy_image and image_source_path.exists():
                #     image_destination_path = export_dir / image_name
                #     shutil.copyfile(image_source_path, image_destination_path)

                data_dict_cpu = {}

                for key, value in data_dict.items():
                    if key == 'points':
                        data_dict_cpu[key] = value.cpu().numpy()

                with open(f'{export_dir}/data_dict_{idx+1:06d}.pkl', 'wb') as f:
                    pickle.dump(data_dict_cpu, f, pickle.HIGHEST_PROTOCOL)

                for pred_dict in pred_dicts:
                    for key, value in pred_dict.items():
                        if isinstance(pred_dict[key], torch.Tensor):
                            pred_dict[key] = value.cpu().numpy()

                with open(f'{export_dir}/pred_dicts_{idx + 1:06d}.pkl', 'wb') as f:
                    pickle.dump(pred_dicts, f, pickle.HIGHEST_PROTOCOL)
            else:
                V.draw_scenes(
                    points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
                )
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
