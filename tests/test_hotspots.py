import numpy as np

from pvdet.hotspots import detect


def test_detects_hot_spot_on_uniform_panel():
    temp = np.full((100, 100), 35.0, np.float32)
    temp[40:46, 60:66] = 45.0  # 6×6 px, +10 °C
    temp[:, 20] = 25.0  # zimna przerwa między panelami — nie może dać wykrycia
    spots = detect(temp)
    assert len(spots) == 1
    assert spots[0]["bbox"] == (60, 40, 65, 45) and spots[0]["type"] == "hotspot"
    assert abs(spots[0]["delta_c"] - 10) < 0.01 and spots[0]["temp_c"] == 45.0
