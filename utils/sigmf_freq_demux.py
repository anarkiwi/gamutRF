#!/usr/bin/env python3

from collections import defaultdict
import numpy as np
import os
import sys
from tqdm import tqdm
from sigmf import SigMFFile, sigmffile
from sigmf.utils import get_data_type_str

ANNOTATE_SCAN = True
FILTER_ZEROS = True
ZERO_BLOCK_SIZE = 1024


def filter_leading_annotation_zeros(samples, annotations, block_size=1024):
    """
    For each annotation, remove leading zero samples (in multiples of block_size)
    from the beginning of that annotation's region. Zeros outside annotation regions,
    or not at the start of an annotation, are left in place.
    Annotations reduced to zero length after filtering are dropped.
    Returns (filtered_samples, adjusted_annotations).
    """
    if not annotations:
        return samples, []

    keep = np.ones(len(samples), dtype=bool)

    for start, length, _metadata in annotations:
        if start >= len(samples) or length == 0:
            continue
        end = min(start + length, len(samples))
        nonzero_indices = np.nonzero(samples[start:end])[0]
        leading_zeros = (
            int(nonzero_indices[0]) if len(nonzero_indices) > 0 else (end - start)
        )
        remove_count = (leading_zeros // block_size) * block_size
        if remove_count > 0:
            keep[start : start + remove_count] = False

    filtered = samples[keep]

    kept_before = np.zeros(len(samples) + 1, dtype=np.intp)
    kept_before[1:] = np.cumsum(keep)

    adjusted = []
    for start, length, metadata in annotations:
        s = min(start, len(samples))
        e = min(start + length, len(samples))
        new_start = int(kept_before[s])
        new_length = int(kept_before[e] - kept_before[s])
        if new_length > 0 or length == 0:
            adjusted.append((new_start, new_length, metadata))

    return filtered, adjusted


def demux_file(
    path,
    annotate_scan=ANNOTATE_SCAN,
    filter_zeros=FILTER_ZEROS,
    block_size=ZERO_BLOCK_SIZE,
):
    if not path.endswith("sigmf-meta"):
        raise ValueError(path)
    signal = sigmffile.fromfile(path)

    sample_rate = signal.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
    sample_count = signal.sample_count
    captures = signal.get_captures()
    # workaround incorrect first tuning interval.
    captures = captures[1:]
    annotations = signal.get_annotations()

    freq_files = {}
    freq_annotations = defaultdict(list)
    freq_samples_written = defaultdict(int)
    freq_captures = defaultdict(int)

    print(f"demuxing {path}")
    for adx, capture in enumerate(tqdm(captures)):
        freq_center = int(capture.get(SigMFFile.FREQUENCY_KEY, 0))
        capture_start_idx = capture[SigMFFile.START_INDEX_KEY]
        try:
            next_capture = captures[adx + 1]
            next_capture_start_idx = next_capture[SigMFFile.START_INDEX_KEY]
        except IndexError:
            next_capture_start_idx = sample_count
        capture_length = next_capture_start_idx - capture_start_idx
        samples = signal.read_samples(capture_start_idx, capture_length)
        capture_annotations = [
            annotation
            for annotation in annotations
            if annotation[SigMFFile.START_INDEX_KEY] > capture_start_idx
            and annotation[SigMFFile.START_INDEX_KEY]
            <= (capture_start_idx + capture_length)
        ]

        rel_annotations = [
            (
                annotation[SigMFFile.START_INDEX_KEY] - capture_start_idx,
                annotation[SigMFFile.LENGTH_INDEX_KEY],
                {
                    k: v
                    for k, v in annotation.items()
                    if not k.startswith("core:sample_")
                },
            )
            for annotation in capture_annotations
        ]

        if filter_zeros:
            samples, rel_annotations = filter_leading_annotation_zeros(
                samples, rel_annotations, block_size
            )

        capture_length = len(samples)

        if freq_center in freq_files:
            _data_type, data_file = freq_files[freq_center]
            with open(data_file, "ba") as f:
                samples.tofile(f)
        else:
            data_file = os.path.join(
                os.path.dirname(path),
                f"{int(round(freq_center/1e6))}MHz-demux-{os.path.splitext(os.path.basename(path))[0]}.sigmf-data",
            )
            data_type = get_data_type_str(samples)
            freq_files[freq_center] = (data_type, data_file)
            samples.tofile(data_file)

        if annotate_scan:
            freq_annotations[freq_center].append(
                (
                    freq_samples_written[freq_center],
                    1024,
                    {
                        "core:label": f"SCAN {freq_captures[freq_center]}",
                        "core:freq_lower_edge": freq_center - sample_rate / 10,
                        "core:freq_upper_edge": freq_center + sample_rate / 10,
                    },
                )
            )

        for rel_start, length, metadata in rel_annotations:
            freq_annotations[freq_center].append(
                (rel_start + freq_samples_written[freq_center], length, metadata)
            )

        freq_samples_written[freq_center] += capture_length
        freq_captures[freq_center] += 1

    print("writing demuxed metas")
    for freq_center, (data_type, data_file) in tqdm(freq_files.items()):
        meta = SigMFFile(
            data_file=data_file,
            global_info={
                SigMFFile.DATATYPE_KEY: data_type,
                SigMFFile.SAMPLE_RATE_KEY: sample_rate,
            },
        )
        meta.add_capture(
            0,
            metadata={
                SigMFFile.FREQUENCY_KEY: freq_center,
            },
        )
        for sample_start, sample_count, metadata in freq_annotations[freq_center]:
            meta.add_annotation(sample_start, sample_count, metadata=metadata)
        meta.tofile(os.path.splitext(data_file)[0] + ".sigmf-meta")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        demux_file(path)
