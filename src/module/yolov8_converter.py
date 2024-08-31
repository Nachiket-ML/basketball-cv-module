"""YOLOv8Pose postprocessing (converter)."""

from typing import Any, Tuple

import numpy as np
from numba.typed import List

from savant.base.converter import BaseObjectModelOutputConverter
from savant.base.model import ObjectModel
from savant.utils.nms import nms_cpu, nms_gpu


class YoloV8ObjectConverter(BaseObjectModelOutputConverter):
    """`YOLOv8 converter."""

    def __init__(
        self,
        confidence_threshold: float = 0.1,
        nms_iou_threshold: float = 0.7,
        top_k: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.top_k = top_k

    def __call__(
        self,
        *output_layers: np.ndarray,
        model: ObjectModel,
        roi: Tuple[float, float, float, float],
    ) -> Tuple[np.ndarray, List[List[Tuple[str, Any, float]]]]:
        """Converts output layer tensors to bboxes and key points.

        :param output_layers: Output layers tensor
        :param model: Model definition, required parameters: input tensor shape
        :param roi: [top, left, width, height] of the rectangle
            on which the model infers
        :return: a combination of :py:class:`.BaseObjectModelOutputConverter` and
            :py:class:`.BaseAttributeModelOutputConverter` outputs:

            * BBox tensor ``(class_id, confidence, xc, yc, width, height, [angle])``
              offset by roi upper left and scaled by roi width and height,
            * list of attributes values with confidences
              ``(attr_name, value, confidence)``
        TODO: Set tensor_format to CuPy to load output_layers onto GPU.
        TODO: Change nms_cpu to nms_gpu
        """
        # print(f'dets: {output_layers[0]}\n')
        # output_layers[0]: (100,5) where each row is bbox. Columns: x_center, y_center, width, height, confidence
        # output_layers[1]: (100,) where each value is a class id
        # print(f'dets shape: {output_layers[0].shape}\n') 
        # print(f'labels: {output_layers[1]}\n')
        # print(f'labels shape: {output_layers[1].shape}\n') # (100,)
        
        output = output_layers[0]
        labels = output_layers[1]

        confidences = output[:, 4] 
        # print(f'confidences: {confidences}\n')
        keep = confidences > self.confidence_threshold
        print(f'keep: {keep}')
        if keep.any():
            print(f'keep shape: {keep.shape}')

        if not keep.any():
            print('return empty tensor')
            # return ret_empty
            return np.zeros(output.shape)

        output = output[keep]
        labels = labels[keep]
        bboxes = output[:, :4] # (x_left, y_top, x_right, y_bottom)
        confidences = output[:, 4]
        print(f'nms boxes: {bboxes}')
        print(f'confidences: {confidences}')

        # Convert to bboxes to (xc, yc, width, height) format
        width = np.subtract(bboxes[:, 2], bboxes[:, 0])
        height = np.subtract(bboxes[:, 3], bboxes[:, 1])
        xc = np.add(bboxes[:, 0], width/2).reshape((bboxes.shape[0],1))
        yc = np.add(bboxes[:, 1], height/2).reshape((bboxes.shape[0],1))
        width = width.reshape((bboxes.shape[0],1))
        height = height.reshape((bboxes.shape[0],1))
        bboxes = np.concatenate((xc, yc, width, height), axis=1)
        print(f'bboxes.shape: {bboxes.shape}')

        output[:, :4] = bboxes
        keep = nms_cpu(
            bboxes,
            confidences,
            self.nms_iou_threshold,
            self.top_k,
        )
        print(f'keep 1: {keep}')
        if keep.any():
            print(f'keep shape: {keep.shape}')
        if not keep.any():
            print('return empty tensor 1')
            # return ret_empty
            return np.zeros(output.shape)

        output = output[keep]
        print(f'filtered output: {output}\n filtered output.shape: {output.shape}\n') 

        labels = labels[keep].reshape((bboxes.shape[0], 1)).astype(np.uint16)
        print(f'filtered labels: {labels}\n filtered labels.shape {labels.shape}\n')
        
        confidences = output[:, 4:5]
        print(f'confidences: {confidences}\n confidences.shape: {confidences.shape}')
        bboxes = output[:, :4]

        roi_left, roi_top, roi_width, roi_height = roi
        print(f'roi_left: {roi_left}\n roi_top: {roi_top}\n roi_width: {roi_width}\n roi_height: {roi_height}\n')
        if model.input.maintain_aspect_ratio:
            ratio_x = ratio_y = min(
                model.input.width / roi_width,
                model.input.height / roi_height,
            )
        else:
            ratio_x = model.input.width / roi_width
            ratio_y = model.input.height / roi_height
        print(f'ratio_x: {ratio_x}\n ratio_y: {ratio_y}')

        bboxes /= np.float32([ratio_x, ratio_y, ratio_x, ratio_y])
        bboxes[:, 0] += roi_left
        bboxes[:, 1] += roi_top

        print(f'bboxes: {bboxes}')
        if bboxes.shape:
            print(f'bboxes.shape {bboxes.shape}')
        else:
            print(f'bboxes has no shape')

        bboxes = np.concatenate((labels, confidences, bboxes), axis=1)
        return bboxes
