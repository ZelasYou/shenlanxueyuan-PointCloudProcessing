#create_pred_from_ground_truth.PY
#!/opt/conda/envs/deep-detection/bin/python

import argparse

import os
import glob

import pandas as pd

import progressbar


def generate_detection_results(input_dir, output_dir):
    """
    Create KITTI 3D object detection results from labels

    """
    # create output dir:
    os.mkdir(
        os.path.join(output_dir, 'data')
    )

    # get input point cloud filename:
    for input_filename in progressbar.progressbar(
        glob.glob(
            os.path.join(input_dir, '*.txt')
        )
    ):
        # read data:
        label = pd.read_csv(input_filename, sep=' ', header=None)
        label.columns = [
            'category',
            'truncation', 'occlusion',
            'alpha',
            '2d_bbox_left', '2d_bbox_top', '2d_bbox_right', '2d_bbox_bottom',
            'height', 'width', 'length',
            'location_x', 'location_y', 'location_z',
            'rotation'
        ]
        # add score:
        label['score'] = 100.0
        # create output:
        output_filename = os.path.join(
            output_dir, 'data', os.path.basename(input_filename)
        )
        label.to_csv(output_filename, sep=' ', header=False, index=False)

def get_arguments():
    """
    Get command-line arguments

    """
    # init parser:
    parser = argparse.ArgumentParser("Generate KITTI 3D Object Detection result from ground truth labels.")

    # add required and optional groups:
    required = parser.add_argument_group('Required')

    # add required:
    required.add_argument(
        "-i", dest="input", help="Input path of ground truth labels.",
        required=True, type=str
    )
    required.add_argument(
        "-o", dest="output", help="Output path of detection results.",
        required=True, type=str
    )

    # parse arguments:
    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments:
    arguments = get_arguments()

    generate_detection_results(arguments.input, arguments.output)
