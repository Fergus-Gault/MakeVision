from enum import Enum
import cv2
import numpy as np
from typing import Dict


class ColorSpace(str, Enum):
    """Enum representing different color spaces."""
    RGB = "RGB"
    BGR = "BGR"
    GRAY = "GRAY"
    HSV = "HSV"
    LAB = "LAB"
    YUV = "YUV"
    RGBA = "RGBA"
    BGRA = "BGRA"

    def __str__(self) -> str:
        return self.value


class FrameData:
    def __init__(self, frame: np.ndarray, color_space: ColorSpace, frames: Dict[str, np.ndarray] = None) -> None:
        self._frame = frame
        self._color_space = color_space
        self._frames = frames

    @property
    def frame(self) -> np.ndarray:
        """Get the frame data."""
        return self._frame

    @frame.setter
    def frame(self, value: np.ndarray):
        self._frame = value

    @property
    def frames(self) -> Dict[str, np.ndarray]:
        """Get the frames dictionary for multi-camera setups."""
        return self._frames

    @frames.setter
    def frames(self, value: Dict[str, np.ndarray]):
        """Set the frames dictionary for multi-camera setups."""
        self._frames = value

    @property
    def color_space(self) -> ColorSpace:
        """Get the color space of the frame."""
        return self._color_space

    @color_space.setter
    def color_space(self, value: ColorSpace):
        # Convert to the requested color space
        if self._color_space == value:
            return

        if value == ColorSpace.HSV:
            self._color_to_hsv()
        elif value == ColorSpace.RGB:
            self._color_to_rgb()
        elif value == ColorSpace.BGR:
            self._color_to_bgr()
        elif value == ColorSpace.GRAY:
            self._color_to_gray()
        elif value == ColorSpace.LAB:
            self._color_to_lab()
        elif value == ColorSpace.YUV:
            self._color_to_yuv()
        elif value == ColorSpace.RGBA:
            self._color_to_rgba()
        elif value == ColorSpace.BGRA:
            self._color_to_bgra()
        else:
            raise ValueError(f"Unsupported color space: {value}")

        self._color_space = value

    def _color_to_hsv(self) -> None:
        """Convert the frame to HSV color space."""
        if self._color_space == ColorSpace.HSV:
            return
        self._convert_color_space(ColorSpace.HSV)

    def _color_to_rgb(self) -> None:
        """Convert the frame to RGB color space."""
        if self._color_space == ColorSpace.RGB:
            return
        self._convert_color_space(ColorSpace.RGB)

    def _color_to_bgr(self) -> None:
        """Convert the frame to BGR color space."""
        if self._color_space == ColorSpace.BGR:
            return
        self._convert_color_space(ColorSpace.BGR)

    def _color_to_gray(self) -> None:
        """Convert the frame to grayscale."""
        if self._color_space == ColorSpace.GRAY:
            return
        self._convert_color_space(ColorSpace.GRAY)

    def _color_to_lab(self) -> None:
        """Convert the frame to LAB color space."""
        if self._color_space == ColorSpace.LAB:
            return
        self._convert_color_space(ColorSpace.LAB)

    def _color_to_yuv(self) -> None:
        """Convert the frame to YUV color space."""
        if self._color_space == ColorSpace.YUV:
            return
        self._convert_color_space(ColorSpace.YUV)

    def _color_to_rgba(self) -> None:
        """Convert the frame to RGBA color space."""
        if self._color_space == ColorSpace.RGBA:
            return
        self._convert_color_space(ColorSpace.RGBA)

    def _color_to_bgra(self) -> None:
        """Convert the frame to BGRA color space."""
        if self._color_space == ColorSpace.BGRA:
            return
        self._convert_color_space(ColorSpace.BGRA)

    def _convert_color_space(self, target: ColorSpace) -> None:
        """Convert the frame to the target color space."""
        # Define conversion mappings for each source->target combination
        conversion_map = {
            # From BGR to other spaces
            (ColorSpace.BGR, ColorSpace.RGB): cv2.COLOR_BGR2RGB,
            (ColorSpace.BGR, ColorSpace.GRAY): cv2.COLOR_BGR2GRAY,
            (ColorSpace.BGR, ColorSpace.HSV): cv2.COLOR_BGR2HSV,
            (ColorSpace.BGR, ColorSpace.LAB): cv2.COLOR_BGR2LAB,
            (ColorSpace.BGR, ColorSpace.YUV): cv2.COLOR_BGR2YUV,
            (ColorSpace.BGR, ColorSpace.RGBA): cv2.COLOR_BGR2RGBA,
            (ColorSpace.BGR, ColorSpace.BGRA): cv2.COLOR_BGR2BGRA,

            # From RGB to other spaces
            (ColorSpace.RGB, ColorSpace.BGR): cv2.COLOR_RGB2BGR,
            (ColorSpace.RGB, ColorSpace.GRAY): cv2.COLOR_RGB2GRAY,
            (ColorSpace.RGB, ColorSpace.HSV): cv2.COLOR_RGB2HSV,
            (ColorSpace.RGB, ColorSpace.LAB): cv2.COLOR_RGB2LAB,
            (ColorSpace.RGB, ColorSpace.YUV): cv2.COLOR_RGB2YUV,
            (ColorSpace.RGB, ColorSpace.RGBA): cv2.COLOR_RGB2RGBA,
            (ColorSpace.RGB, ColorSpace.BGRA): cv2.COLOR_RGB2BGRA,

            # From GRAY to other spaces
            (ColorSpace.GRAY, ColorSpace.BGR): cv2.COLOR_GRAY2BGR,
            (ColorSpace.GRAY, ColorSpace.RGB): cv2.COLOR_GRAY2RGB,
            (ColorSpace.GRAY, ColorSpace.HSV): lambda img: cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV),
            (ColorSpace.GRAY, ColorSpace.LAB): lambda img: cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB),
            (ColorSpace.GRAY, ColorSpace.YUV): lambda img: cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2YUV),
            (ColorSpace.GRAY, ColorSpace.RGBA): lambda img: cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGBA),
            (ColorSpace.GRAY, ColorSpace.BGRA): lambda img: cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2BGRA),

            # From HSV to other spaces
            (ColorSpace.HSV, ColorSpace.BGR): cv2.COLOR_HSV2BGR,
            (ColorSpace.HSV, ColorSpace.RGB): cv2.COLOR_HSV2RGB,
            (ColorSpace.HSV, ColorSpace.GRAY): lambda img: cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY),
            (ColorSpace.HSV, ColorSpace.LAB): lambda img: cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2LAB),
            (ColorSpace.HSV, ColorSpace.YUV): lambda img: cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2YUV),
            (ColorSpace.HSV, ColorSpace.RGBA): lambda img: cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2RGBA),
            (ColorSpace.HSV, ColorSpace.BGRA): lambda img: cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2BGRA),

            # From LAB to other spaces
            (ColorSpace.LAB, ColorSpace.BGR): cv2.COLOR_LAB2BGR,
            (ColorSpace.LAB, ColorSpace.RGB): cv2.COLOR_LAB2RGB,
            (ColorSpace.LAB, ColorSpace.GRAY): lambda img: cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2GRAY),
            (ColorSpace.LAB, ColorSpace.HSV): lambda img: cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2HSV),
            (ColorSpace.LAB, ColorSpace.YUV): lambda img: cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2YUV),
            (ColorSpace.LAB, ColorSpace.RGBA): lambda img: cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGBA),
            (ColorSpace.LAB, ColorSpace.BGRA): lambda img: cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2BGRA),

            # From YUV to other spaces
            (ColorSpace.YUV, ColorSpace.BGR): cv2.COLOR_YUV2BGR,
            (ColorSpace.YUV, ColorSpace.RGB): cv2.COLOR_YUV2RGB,
            (ColorSpace.YUV, ColorSpace.GRAY): lambda img: cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_YUV2BGR), cv2.COLOR_BGR2GRAY),
            (ColorSpace.YUV, ColorSpace.HSV): lambda img: cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_YUV2BGR), cv2.COLOR_BGR2HSV),
            (ColorSpace.YUV, ColorSpace.LAB): lambda img: cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_YUV2BGR), cv2.COLOR_BGR2LAB),
            (ColorSpace.YUV, ColorSpace.RGBA): lambda img: cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_YUV2BGR), cv2.COLOR_BGR2RGBA),
            (ColorSpace.YUV, ColorSpace.BGRA): lambda img: cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_YUV2BGR), cv2.COLOR_BGR2BGRA),

            # From RGBA to other spaces
            (ColorSpace.RGBA, ColorSpace.BGR): cv2.COLOR_RGBA2BGR,
            (ColorSpace.RGBA, ColorSpace.RGB): cv2.COLOR_RGBA2RGB,
            (ColorSpace.RGBA, ColorSpace.GRAY): cv2.COLOR_RGBA2GRAY,
            (ColorSpace.RGBA, ColorSpace.HSV): lambda img: cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGBA2BGR), cv2.COLOR_BGR2HSV),
            (ColorSpace.RGBA, ColorSpace.LAB): lambda img: cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGBA2BGR), cv2.COLOR_BGR2LAB),
            (ColorSpace.RGBA, ColorSpace.YUV): lambda img: cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGBA2BGR), cv2.COLOR_BGR2YUV),
            (ColorSpace.RGBA, ColorSpace.BGRA): cv2.COLOR_RGBA2BGRA,

            # From BGRA to other spaces
            (ColorSpace.BGRA, ColorSpace.BGR): cv2.COLOR_BGRA2BGR,
            (ColorSpace.BGRA, ColorSpace.RGB): cv2.COLOR_BGRA2RGB,
            (ColorSpace.BGRA, ColorSpace.GRAY): cv2.COLOR_BGRA2GRAY,
            (ColorSpace.BGRA, ColorSpace.HSV): lambda img: cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2HSV),
            (ColorSpace.BGRA, ColorSpace.LAB): lambda img: cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2LAB),
            (ColorSpace.BGRA, ColorSpace.YUV): lambda img: cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2YUV),
            (ColorSpace.BGRA, ColorSpace.RGBA): cv2.COLOR_BGRA2RGBA,
        }

        conversion_key = (self._color_space, target)
        if conversion_key in conversion_map:
            converter = conversion_map[conversion_key]
            if callable(converter):
                self._frame = converter(self._frame)
            else:
                self._frame = cv2.cvtColor(self._frame, converter)
        else:
            raise ValueError(
                f"Cannot convert from {self._color_space} to {target}.")

        self._color_space = target
