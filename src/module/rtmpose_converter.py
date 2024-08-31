"""YOLOv8Pose postprocessing (converter)."""

from typing import Any, Tuple

import numpy as np
from numba.typed import List

from savant.base.converter import BaseAttributeModelOutputConverter
from savant.base.model import AttributeModel
from savant.utils.nms import nms_cpu


class RTMPoseConverter(BaseAttributeModelOutputConverter):
    """RTMPose converter.
    TODO: Set tensor_format to CuPy to load output_layers onto GPU.
    Link: https://docs.savant-ai.io/develop/reference/api/generated/savant.base.converter.BaseAttributeModelOutputConverter.html
    """
    def __init__(
        self,
        confidence_threshold: float = 0.6,
        nms_iou_threshold: float = 0.45,
        top_k: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.top_k = top_k
        # Values come from rtmpose pipeline.json
        self.sigma_x = 4.9
        self.sigma_y = 5.66
        self.simcc_split_ratio = 2.0
        self.padding = 1.25
        self.input_width = 192
        self.input_height = 256

    def __call__(
        self,
        *output_layers: np.ndarray,
        model: AttributeModel,
        roi: Tuple[float, float, float, float],
    ) -> Tuple[np.ndarray, List[List[Tuple[str, Any, float]]]]:
        """Converts output layer tensors to bboxes and key points.

        :param output_layers: Numpy array with two values: simcc_x and simcc_y, each
         with n rows (n = 17 in simcc format). Simcc_x has simcc_split_ratio*input.shape[2]
         columns (2*192 = 384) and simcc_y has simcc_split_ratio*input.shape[1] columns (2*256 = 512).
        :param model: Model definition, required parameters: input tensor shape
        :param roi: [top, left, width, height] of the bounding box around a person object
            on which the model infers
        :return: A list of n keypoints in (x,y) coordinate format
        """
        top, left, width, height = roi
        simcc_x, simcc_y = output_layers
        num_keypoints = simcc_x.shape[0]
        print(f'ROI top, left, width, height: {top}, {left}, {width}, {height}') # This should be the bounding box coordinates of a person object.
        print(f'simcc_x: {simcc_x}\n simcc_x.shape: {simcc_x.shape}')
        print(f'simcc_y: {simcc_y}\n simcc_y.shape: {simcc_y.shape}')

        # Select pixel (column) with largest confidences for each keypoint
        max_indices_simcc_x = np.argmax(simcc_x, axis=1)
        max_indices_simcc_y = np.argmax(simcc_y, axis=1)

        # Scale down pixel indices to match the rtmpose input shape (192, 256)
        max_indices_simcc_x = np.divide(max_indices_simcc_x, self.simcc_split_ratio)
        max_indices_simcc_y = np.divide(max_indices_simcc_y, self.simcc_split_ratio)

        # Scale down indices again to match ROI shape (width, height) and shift to (top, left)
        max_indices_simcc_x = left + max_indices_simcc_x*(width/self.input_width)
        max_indices_simcc_y = top + max_indices_simcc_y*(height/self.input_height)
        max_indices_simcc_x = max_indices_simcc_x.reshape((num_keypoints, 1)).astype(np.uint16)
        max_indices_simcc_y = max_indices_simcc_y.reshape((num_keypoints, 1)).astype(np.uint16)

        keypoint_labels = np.array([f'keypoint_{i}' for i in range(num_keypoints)]).reshape((num_keypoints, 1))
        keypoints = np.concatenate((keypoint_labels, max_indices_simcc_x, max_indices_simcc_y), axis=1)
        print(f'Keypoints RTMPose converter: {keypoints}')
        return keypoints
        # output = np.transpose(output_layers[0])

        # ret_empty = np.float32([]), []

        # confidences = output[:, 4]
        # keep = confidences > self.confidence_threshold
        # if not keep.any():
        #     return ret_empty

        # output = output[keep]

        
        # confidences = output[:, 4]
        # if not keep.any():
        #     return ret_empty

        # output = output[keep]

        # # person class id = 0
        # class_ids = np.zeros((output.shape[0], 1), dtype=np.float32)
        # confidences = output[:, 4:5]
        # key_points = output[:, 5:]
        # key_points = key_points.reshape(key_points.shape[0], -1, 3)

        # # scale
        # roi_left, roi_top, roi_width, roi_height = roi
        # if model.input.maintain_aspect_ratio:
        #     ratio_x = ratio_y = min(
        #         model.input.width / roi_width,
        #         model.input.height / roi_height,
        #     )
        # else:
        #     ratio_x = model.input.width / roi_width
        #     ratio_y = model.input.height / roi_height

        # key_points /= np.float32([ratio_x, ratio_y, 1.0])
        # key_points[:, 0] += roi_left
        # key_points[:, 1] += roi_top


        # attr_name = model.output.attributes[0].name
        # key_points = [[(attr_name, pts, 1.0)] for pts in key_points]

