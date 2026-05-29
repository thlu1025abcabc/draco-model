from __future__ import annotations

from draco_model.market.minute_calendar import MinuteCalendar


def test_minute_calendar_caches_bucket_map_by_interval() -> None:
    calendar = MinuteCalendar()

    first = calendar.bucket_map(5).collect()
    second = calendar.bucket_map(5).collect()

    assert list(calendar._bucket_maps) == [5]
    assert first.equals(second)
    assert first.filter(first["minute"] == 930)["__bucket_minute"].to_list() == [930]
    assert first.filter(first["minute"] == 934)["__bucket_minute"].to_list() == [930]
    assert first.filter(first["minute"] == 935)["__bucket_minute"].to_list() == [935]
