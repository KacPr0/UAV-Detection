import numpy as np

from pvdet.yolo import encode


def test_encode_marks_hot_spot_in_excess_channel():
    temp = np.full((64, 64), 35.0, np.float32)
    temp[30:34, 30:34] = 45.0
    img = encode(temp)
    assert img.shape == (64, 64, 3) and img.dtype == np.uint8
    assert img[32, 32, 1] == 255 and img[5, 5, 1] == 0  # +10 °C nad tłem → pełna skala, tło → 0
