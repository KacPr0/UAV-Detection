import numpy as np

from pvdet.io.radiometric import raw_to_celsius

XT2 = dict(r1=348025, r2=1, b=1428, f=1, o=-425)  # stałe z plików zbioru hf_thermal


def test_raw_to_celsius_plausible_for_panels():
    t = raw_to_celsius(np.array([3166, 4635]), **XT2)
    assert 15 < t[0] < 30 and 40 < t[1] < 60
