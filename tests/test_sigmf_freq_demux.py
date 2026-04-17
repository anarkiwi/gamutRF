#!/usr/bin/python3

import importlib.util
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "sigmf_freq_demux",
    os.path.join(os.path.dirname(__file__), "..", "utils", "sigmf_freq_demux.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

filter_leading_annotation_zeros = _mod.filter_leading_annotation_zeros
demux_file = _mod.demux_file


# --- filter_leading_annotation_zeros ---


def test_no_annotations_returns_samples_unchanged():
    samples = np.zeros(2048, dtype=complex)
    out_s, out_a = filter_leading_annotation_zeros(samples, [])
    np.testing.assert_array_equal(out_s, samples)
    assert out_a == []


def test_no_leading_zeros_in_annotation_unchanged():
    samples = np.ones(2048, dtype=complex)
    anns = [(0, 2048, {"label": "sig"})]
    out_s, out_a = filter_leading_annotation_zeros(samples, anns)
    np.testing.assert_array_equal(out_s, samples)
    assert out_a == [(0, 2048, {"label": "sig"})]


def test_leading_zeros_in_annotation_exact_multiple_removed():
    # annotation covers entire buffer: [0,1024) zeros then [1024,2048) real
    samples = np.zeros(2048, dtype=complex)
    samples[1024:] = 1 + 1j
    anns = [(0, 2048, {"label": "sig"})]
    out_s, out_a = filter_leading_annotation_zeros(samples, anns)
    assert len(out_s) == 1024
    np.testing.assert_array_equal(out_s, np.full(1024, 1 + 1j, dtype=complex))
    assert out_a == [(0, 1024, {"label": "sig"})]


def test_leading_zeros_floored_to_block_size():
    # 1500 leading zeros → floor to 1024 removed, 476 zeros remain
    samples = np.zeros(3072, dtype=complex)
    samples[1500:] = 1 + 1j
    anns = [(0, 3072, {"label": "sig"})]
    out_s, out_a = filter_leading_annotation_zeros(samples, anns)
    assert len(out_s) == 3072 - 1024
    assert out_a[0] == (0, 3072 - 1024, {"label": "sig"})


def test_leading_zeros_smaller_than_block_size_not_removed():
    samples = np.zeros(2048, dtype=complex)
    samples[500:] = 1 + 1j
    anns = [(0, 2048, {"label": "sig"})]
    out_s, out_a = filter_leading_annotation_zeros(samples, anns)
    # 500 leading zeros < 1024 block → nothing removed
    np.testing.assert_array_equal(out_s, samples)
    assert out_a == [(0, 2048, {"label": "sig"})]


def test_zeros_before_annotation_start_not_removed():
    # zeros at [0,1024), annotation starts at 1024 (non-zero)
    samples = np.zeros(2048, dtype=complex)
    samples[1024:] = 1 + 1j
    anns = [(1024, 1024, {"label": "sig"})]
    out_s, out_a = filter_leading_annotation_zeros(samples, anns)
    # annotation starts on real data → no leading zeros to remove
    np.testing.assert_array_equal(out_s, samples)
    assert out_a == [(1024, 1024, {"label": "sig"})]


def test_zeros_after_annotation_start_not_removed():
    # annotation: real then zeros at the end
    samples = np.ones(2048, dtype=complex)
    samples[1024:] = 0j
    anns = [(0, 2048, {"label": "sig"})]
    out_s, out_a = filter_leading_annotation_zeros(samples, anns)
    # zeros are at the tail, not the start of the annotation → not removed
    np.testing.assert_array_equal(out_s, samples)
    assert out_a == [(0, 2048, {"label": "sig"})]


def test_zeros_in_middle_of_annotation_not_removed():
    # real | zeros | real, annotation covers all
    samples = np.ones(3072, dtype=complex)
    samples[1024:2048] = 0j
    anns = [(0, 3072, {"label": "sig"})]
    out_s, out_a = filter_leading_annotation_zeros(samples, anns)
    # zeros are mid-annotation, not leading → not removed
    np.testing.assert_array_equal(out_s, samples)
    assert out_a == [(0, 3072, {"label": "sig"})]


def test_annotation_entirely_zeros_dropped():
    samples = np.zeros(2048, dtype=complex)
    anns = [(0, 2048, {"label": "noise"})]
    out_s, out_a = filter_leading_annotation_zeros(samples, anns)
    assert len(out_s) == 0
    assert out_a == []


def test_multiple_annotations_each_filtered_independently():
    # ann1: [0,2048) — 1024 leading zeros → 1024 removed
    # ann2: [2048,4096) — starts on real data → unchanged
    samples = np.zeros(4096, dtype=complex)
    samples[1024:2048] = 1 + 1j
    samples[2048:] = 2 + 2j
    anns = [
        (0, 2048, {"label": "a"}),
        (2048, 2048, {"label": "b"}),
    ]
    out_s, out_a = filter_leading_annotation_zeros(samples, anns)
    assert len(out_s) == 4096 - 1024
    assert out_a[0] == (0, 1024, {"label": "a"})
    # ann2 original start 2048 shifts left by 1024 removed samples
    assert out_a[1] == (1024, 2048, {"label": "b"})


def test_second_annotation_start_shifted_by_first_removal():
    # ann1 removes 1024 leading zeros; ann2 is after → its start shifts left
    samples = np.zeros(4096, dtype=complex)
    samples[1024:] = 1 + 1j
    anns = [
        (0, 1024, {"label": "a"}),  # entirely zeros → dropped
        (2048, 1024, {"label": "b"}),  # starts at real data, shifts left by 1024
    ]
    out_s, out_a = filter_leading_annotation_zeros(samples, anns)
    assert len(out_s) == 4096 - 1024
    assert len(out_a) == 1
    assert out_a[0] == (1024, 1024, {"label": "b"})


def test_zero_length_annotation_preserved_with_adjusted_start():
    samples = np.zeros(2048, dtype=complex)
    samples[1024:] = 1 + 1j
    # zero-length marker annotation after the leading zeros of a prior annotation
    anns = [
        (0, 2048, {"label": "sig"}),
        (2048, 0, {"label": "marker"}),
    ]
    out_s, out_a = filter_leading_annotation_zeros(samples, anns)
    assert len(out_s) == 1024
    assert out_a[0] == (0, 1024, {"label": "sig"})
    assert out_a[1][0] == 1024  # marker shifted left
    assert out_a[1][1] == 0


# --- demux_file integration ---

from sigmf import SigMFFile as _RealSigMFFile

_FREQ_KEY = _RealSigMFFile.FREQUENCY_KEY
_START_KEY = _RealSigMFFile.START_INDEX_KEY
_LEN_KEY = _RealSigMFFile.LENGTH_INDEX_KEY
_RATE_KEY = _RealSigMFFile.SAMPLE_RATE_KEY
_TYPE_KEY = _RealSigMFFile.DATATYPE_KEY


def _make_signal(sample_rate, captures, annotations, samples_by_range, sample_count):
    signal = MagicMock()
    signal.get_global_field.return_value = sample_rate
    signal.sample_count = sample_count
    signal.get_captures.return_value = captures
    signal.get_annotations.return_value = annotations
    signal.read_samples.side_effect = lambda s, c: samples_by_range[(s, c)]
    return signal


def _sigmf_class_mock(meta_out):
    """Return a MagicMock for SigMFFile that keeps real string constants intact."""
    mock_class = MagicMock()
    mock_class.return_value = meta_out
    for attr in (
        "FREQUENCY_KEY",
        "START_INDEX_KEY",
        "LENGTH_INDEX_KEY",
        "SAMPLE_RATE_KEY",
        "DATATYPE_KEY",
    ):
        setattr(mock_class, attr, getattr(_RealSigMFFile, attr))
    return mock_class


def test_demux_file_basic_no_zeros(tmp_path):
    path = str(tmp_path / "rec.sigmf-meta")
    open(path, "w").close()

    freq = int(100e6)
    sample_rate = int(1e6)
    cap0 = {_FREQ_KEY: freq, _START_KEY: 0}
    cap1 = {_FREQ_KEY: freq, _START_KEY: 0}
    cap2 = {_FREQ_KEY: freq, _START_KEY: 2048}

    samples_a = np.ones(2048, dtype=np.complex64)
    samples_b = np.ones(2048, dtype=np.complex64) * (2 + 0j)
    samples_by_range = {(0, 2048): samples_a, (2048, 2048): samples_b}

    signal = _make_signal(sample_rate, [cap0, cap1, cap2], [], samples_by_range, 4096)
    meta_out = MagicMock()

    with patch.object(_mod, "sigmffile") as mock_sigmffile, patch.object(
        _mod, "SigMFFile", _sigmf_class_mock(meta_out)
    ):
        mock_sigmffile.fromfile.return_value = signal
        demux_file(path, annotate_scan=False, filter_zeros=False)

    data_file = tmp_path / "100MHz-demux-rec.sigmf-data"
    assert data_file.exists()
    written = np.fromfile(str(data_file), dtype=np.complex64)
    assert len(written) == 4096


def test_demux_file_filter_zeros_removes_annotation_leading_zeros(tmp_path):
    # demux_file filters annotations where START > capture_start_idx (strict), so the
    # earliest an annotation can start is at capture_start_idx + 1 (relative offset 1).
    # We put 1 non-annotation sample at relative offset 0, then 1024 zeros that ARE
    # the annotation's leading zeros, then real data.
    path = str(tmp_path / "rec.sigmf-meta")
    open(path, "w").close()

    freq = int(200e6)
    sample_rate = int(1e6)
    cap0 = {_FREQ_KEY: freq, _START_KEY: 0}  # skipped as first capture
    cap1 = {_FREQ_KEY: freq, _START_KEY: 0}

    # layout (absolute): [0]=real [1:1025]=zeros [1025:4096]=real
    samples = np.ones(4096, dtype=np.complex64)
    samples[1:1025] = 0j
    samples_by_range = {(0, 4096): samples}

    # annotation starts at abs idx 1 (> 0 = capture_start_idx), covering [1:4096)
    ann = {_START_KEY: 1, _LEN_KEY: 4095, "core:label": "sig"}
    signal = _make_signal(sample_rate, [cap0, cap1], [ann], samples_by_range, 4096)
    meta_out = MagicMock()

    with patch.object(_mod, "sigmffile") as mock_sigmffile, patch.object(
        _mod, "SigMFFile", _sigmf_class_mock(meta_out)
    ):
        mock_sigmffile.fromfile.return_value = signal
        demux_file(path, annotate_scan=False, filter_zeros=True)

    data_file = tmp_path / "200MHz-demux-rec.sigmf-data"
    assert data_file.exists()
    written = np.fromfile(str(data_file), dtype=np.complex64)
    # sample[0] (real, outside annotation) kept; 1024 leading zeros of annotation removed;
    # remaining 3071 real samples kept → total 3072
    assert len(written) == 3072
    assert written[0] == 1 + 0j  # sample before annotation: preserved
    assert np.all(written[1:] == 1 + 0j)  # real data after zero removal


def test_demux_file_filter_zeros_adjusts_annotation_length(tmp_path):
    path = str(tmp_path / "rec.sigmf-meta")
    open(path, "w").close()

    freq = int(300e6)
    sample_rate = int(1e6)
    cap0 = {_FREQ_KEY: freq, _START_KEY: 0}
    cap1 = {_FREQ_KEY: freq, _START_KEY: 0}

    # layout: [0]=real [1:1025]=zeros [1025:4096]=real
    samples = np.ones(4096, dtype=np.complex64)
    samples[1:1025] = 0j
    samples_by_range = {(0, 4096): samples}

    # annotation spans [1, 4096) — 1024 leading zeros then 3071 real
    ann = {_START_KEY: 1, _LEN_KEY: 4095, "core:label": "signal"}
    signal = _make_signal(sample_rate, [cap0, cap1], [ann], samples_by_range, 4096)

    recorded_annotations = []
    meta_out = MagicMock()
    meta_out.add_annotation.side_effect = (
        lambda s, l, metadata=None: recorded_annotations.append((s, l, metadata))
    )

    with patch.object(_mod, "sigmffile") as mock_sigmffile, patch.object(
        _mod, "SigMFFile", _sigmf_class_mock(meta_out)
    ):
        mock_sigmffile.fromfile.return_value = signal
        demux_file(path, annotate_scan=False, filter_zeros=True)

    # 1024 leading zeros removed from annotation; annotation length shrinks by 1024
    assert len(recorded_annotations) == 1
    ann_start, ann_len, ann_meta = recorded_annotations[0]
    # relative start was 1; sample[0] kept before it → new_start stays 1
    assert ann_start == 1
    assert ann_len == 3071  # 4095 - 1024 leading zeros removed
    assert ann_meta["core:label"] == "signal"


def test_demux_file_zeros_outside_annotations_not_removed(tmp_path):
    path = str(tmp_path / "rec.sigmf-meta")
    open(path, "w").close()

    freq = int(400e6)
    sample_rate = int(1e6)
    cap0 = {_FREQ_KEY: freq, _START_KEY: 0}
    cap1 = {_FREQ_KEY: freq, _START_KEY: 0}

    # 1024 zeros before the annotation, then 1024 real samples where annotation starts
    samples = np.zeros(2048, dtype=np.complex64)
    samples[1024:] = 1 + 0j
    samples_by_range = {(0, 2048): samples}

    # annotation starts at abs idx 1024 (first real sample, > 0 = capture_start_idx)
    ann = {_START_KEY: 1024, _LEN_KEY: 1024, "core:label": "sig"}
    signal = _make_signal(sample_rate, [cap0, cap1], [ann], samples_by_range, 2048)
    meta_out = MagicMock()

    with patch.object(_mod, "sigmffile") as mock_sigmffile, patch.object(
        _mod, "SigMFFile", _sigmf_class_mock(meta_out)
    ):
        mock_sigmffile.fromfile.return_value = signal
        demux_file(path, annotate_scan=False, filter_zeros=True)

    data_file = tmp_path / "400MHz-demux-rec.sigmf-data"
    assert data_file.exists()
    written = np.fromfile(str(data_file), dtype=np.complex64)
    # annotation starts on real data (no leading zeros) → all 2048 samples preserved
    assert len(written) == 2048
    assert np.all(written[:1024] == 0 + 0j)  # zeros before annotation kept
    assert np.all(written[1024:] == 1 + 0j)
