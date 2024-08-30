"""Custom DrawFunc implementation."""

from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.utils.artist import Artist


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
            elif obj.label == 'ball':
                print(f'Drawing ball')
                artist.add_bbox(obj.bbox, 3, (0, 255, 0, 255)) #Green
            elif obj.label == 'rim':
                print(f'Drawing rim')
                artist.add_bbox(obj.bbox, 3, (255, 0, 0, 255)) #Red
