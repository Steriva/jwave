from unittest.mock import patch

from jwave.acoustics.progress import (
    _format_duration,
    progress_stride,
    report_simulation_progress,
    reset_simulation_progress,
)


def test_progress_stride_matches_interval():
    assert progress_stride(35000, 5.0) == 1750
    assert progress_stride(35000, 1.0) == 350
    assert progress_stride(35000, 20.0) == 1750
    assert progress_stride(10, 5.0) == 2


def test_format_duration():
    assert _format_duration(0.4) == "0s"
    assert _format_duration(45.2) == "45s"
    assert _format_duration(125.0) == "2m 05s"
    assert _format_duration(3725.0) == "1h 02m"


def test_report_filters_by_interval_pct():
    reset_simulation_progress()
    written = []

    def fake_write(fd, data):
        written.append(data.decode())

    with patch("jwave.acoustics.progress.os.write", side_effect=fake_write):
        report_simulation_progress(0, 100, "Wave propagation", 20.0)
        report_simulation_progress(10, 100, "Wave propagation", 20.0)
        report_simulation_progress(19, 100, "Wave propagation", 20.0)
        report_simulation_progress(39, 100, "Wave propagation", 20.0)
        report_simulation_progress(99, 100, "Wave propagation", 20.0)

    assert len(written) == 4
    assert "step 1/100 (1%)" in written[0]
    assert "step 20/100 (20%)" in written[1]
    assert "step 40/100 (40%)" in written[2]
    assert "step 100/100 (100%)" in written[3]


def test_report_includes_eta():
    written = []

    def fake_write(fd, data):
        written.append(data.decode())

    with patch("jwave.acoustics.progress.os.write", side_effect=fake_write):
        with patch(
            "jwave.acoustics.progress.time.perf_counter",
            side_effect=[0.0, 10.0, 300.0, 300.0],
        ):
            reset_simulation_progress()
            report_simulation_progress(0, 100, "Wave propagation", 50.0)
            report_simulation_progress(49, 100, "Wave propagation", 50.0)
            report_simulation_progress(99, 100, "Wave propagation", 50.0)

    assert "ETA 16m 30s" in written[0]
    assert "ETA 5m 00s" in written[1]
    assert "ETA" not in written[2]
