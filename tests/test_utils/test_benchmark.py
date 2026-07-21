"""
Tests for the tglc.utils.benchmark module, which provides wall-clock timing and peak-memory
reporting helpers.
"""

import logging

import pytest

from tglc.utils.benchmark import benchmark_step, format_benchmark_record, get_peak_rss_bytes


def test_get_peak_rss_bytes_is_plausible():
    peak_rss = get_peak_rss_bytes()
    # Guards against unit regressions: a Python process is far above 1MiB but below 1TiB
    assert 2**20 < peak_rss < 2**40


def test_get_peak_rss_bytes_children_is_nonnegative():
    assert get_peak_rss_bytes(children=True) >= 0


def test_format_benchmark_record():
    record = format_benchmark_record("test", stars=15, elapsed_s=1.23456, label="cutout_0_0")
    assert record == "TGLC-BENCH event=test stars=15 elapsed_s=1.235 label=cutout_0_0"


def test_benchmark_step_logs_record(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.INFO, logger="tglc.utils.benchmark"):
        with benchmark_step("test_step"):
            pass

    matching_records = [
        record.message for record in caplog.records if "TGLC-BENCH event=step" in record.message
    ]
    assert len(matching_records) == 1
    assert "step=test_step" in matching_records[0]
    assert "elapsed_s=" in matching_records[0]
    assert "peak_rss_mb=" in matching_records[0]
    assert "children_peak_rss_mb=" in matching_records[0]


def test_benchmark_step_logs_record_on_error(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.INFO, logger="tglc.utils.benchmark"):
        with pytest.raises(RuntimeError, match="oops"):
            with benchmark_step("failing_step"):
                raise RuntimeError("oops")

    assert any("step=failing_step" in record.message for record in caplog.records)
