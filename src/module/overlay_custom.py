"""Custom DrawFunc implementation."""

from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.utils.artist import Artist
from savant.parameter_storage import param_storage

KEYPOINT_DETECTOR = 'rtmpose_body_2d'

skeleton = [
    ([15, 13], (255, 0, 0, 255)),  # left leg
    ([13, 11], (255, 0, 0, 255)),  # left leg
    ([16, 14], (255, 0, 0, 255)),  # right leg
    ([14, 12], (255, 0, 0, 255)),  # right leg
    ([11, 12], (255, 0, 255, 255)),  # body
    ([5, 11], (255, 0, 255, 255)),  # body
    ([6, 12], (255, 0, 255, 255)),  # body
    ([5, 6], (255, 0, 255, 255)),  # body
    ([5, 7], (0, 255, 0, 255)),  # left arm
    ([7, 9], (0, 255, 0, 255)),  # left arm
    ([6, 8], (0, 255, 0, 255)),  # right arm
    ([8, 10], (0, 255, 0, 255)),  # right arm
    ([1, 2], (255, 255, 0, 255)),  # head
    ([0, 1], (255, 255, 0, 255)),  # head
    ([0, 2], (255, 255, 0, 255)),  # head
    ([1, 3], (255, 255, 0, 255)),  # head
    ([2, 4], (255, 255, 0, 255)),  # head
    ([3, 5], (255, 255, 0, 255)),  # head
    ([4, 6], (255, 255, 0, 255)),  # head
]

# key points with the confidence below the threshold won't be displayed
KP_CONFIDENCE_THRESHOLD = 0.4

KP_LABELS = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
]

class Overlay(NvDsDrawFunc):
    """Custom implementation of PyFunc for drawing on frame."""

    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
        """Draws on frame using the artist and the frame's metadata.

        :param frame_meta: Frame metadata.
        :param artist: Artist to draw on the frame.
        """
        # When the dev_mode is enabled in the module config
        # The draw func code changes are applied without restarting the module

        # super().draw_on_frame(frame_meta, artist)

        # for example, draw a white bounding box around persons
        # and a green bounding box around faces
        # TODO: Add textual label for each bounding box (person, ball, rim)
        # TODO: Draw keypoints from rtmpose model output
        for obj in frame_meta.objects:
            if obj.label == 'frame': continue
            # print(f'object label: {obj.label}')
            print(f'object confidence: {obj.confidence}')
            if obj.label == 'person':
                print(f'Drawing person')
                artist.add_bbox(obj.bbox, 3, (255, 255, 255, 255)) #RGBA
                kp_attr = obj.get_attr_meta(KEYPOINT_DETECTOR, 'keypoints')
                print(f'kp_attr: {kp_attr}')
                if not kp_attr:
                    continue
                key_points = kp_attr.value # x,y coordinates
                print(f'kp_attr.value: {key_points}')
                for pair, color in skeleton:
                    artist.add_line(
                        pt1=(
                            int(key_points[pair[0]][0]),
                            int(key_points[pair[0]][1]),
                        ),
                        pt2=(
                            int(key_points[pair[1]][0]),
                            int(key_points[pair[1]][1]),
                        ),
                        color=color,
                        thickness=2,
                    )
                for (x, y) in key_points:
                    # if conf > KP_CONFIDENCE_THRESHOLD:
                    artist.add_circle(
                        center=(int(x), int(y)),
                        radius=2,
                        color=(255, 0, 0, 255),
                        thickness=2,
                    )
            elif obj.label == 'ball':
                print(f'Drawing ball')
                artist.add_bbox(obj.bbox, 3, (0, 255, 0, 255)) #Green
            elif obj.label == 'rim':
                print(f'Drawing rim')
                artist.add_bbox(obj.bbox, 3, (255, 0, 0, 255)) #Red

# FILTER KEYPOINTS BY CONFIDENCE
# for pair, color in skeleton:
#                     if (
#                         key_points[pair[0]][2] > KP_CONFIDENCE_THRESHOLD
#                         and key_points[pair[1]][2] > KP_CONFIDENCE_THRESHOLD
#                     ):
#                         artist.add_line(
#                             pt1=(
#                                 int(key_points[pair[0]][0]),
#                                 int(key_points[pair[0]][1]),
#                             ),
#                             pt2=(
#                                 int(key_points[pair[1]][0]),
#                                 int(key_points[pair[1]][1]),
#                             ),
#                             color=color,
#                             thickness=2,
#                         )
#                 for i, (x, y, conf) in enumerate(key_points):
#                     if conf > KP_CONFIDENCE_THRESHOLD:
#                         artist.add_circle(
#                             center=(int(x), int(y)),
#                             radius=2,
#                             color=(255, 0, 0, 255),
#                             thickness=2,
#                         )