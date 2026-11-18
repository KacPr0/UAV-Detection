from pvdet.panels import assign


def test_assign_uses_spot_center():
    panels = [(0, 0, 49, 29), (50, 0, 99, 29)]
    spots = assign([{"bbox": (60, 10, 64, 14)}, {"bbox": (45, 5, 47, 7)}, {"bbox": (10, 40, 12, 42)}], panels)
    assert [s["panel"] for s in spots] == [2, 1, None]
