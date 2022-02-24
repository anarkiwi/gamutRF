#!/usr/bin/env python3
import argparse
import gzip
import os

import numpy as np
from matplotlib.mlab import detrend
from matplotlib.mlab import detrend_none
from matplotlib.mlab import stride_windows
from matplotlib.mlab import window_hanning
import matplotlib.pyplot as plt
from scipy.fft import fft  # pylint disable=no-name-in-module
from scipy.fft import fftfreq  # pylint disable=no-name-in-module
import zstandard

from gamutrf.utils import replace_ext, parse_filename, get_nondot_files


def spectral_helper(x, NFFT=None, Fs=None, detrend_func=None,
                    window=None, noverlap=None, pad_to=None,
                    scale_by_freq=None, mode=None):
    if Fs is None:
        Fs = 2
    if noverlap is None:
        noverlap = 0
    if detrend_func is None:
        detrend_func = detrend_none
    if window is None:
        window = window_hanning

    # if NFFT is set to None use the whole signal
    if NFFT is None:
        NFFT = 256

    if mode is None or mode == 'default':
        mode = 'psd'

    if pad_to is None:
        pad_to = NFFT

    if mode != 'psd':
        scale_by_freq = False
    elif scale_by_freq is None:
        scale_by_freq = True

    # For real x, ignore the negative frequencies unless told otherwise
    numFreqs = pad_to
    if pad_to % 2:
        freqcenter = (pad_to - 1)//2 + 1
    else:
        freqcenter = pad_to//2
    scaling_factor = 1.

    freqs = fftfreq(pad_to, 1/Fs)[:numFreqs]
    # center the frequency range at zero
    freqs = np.roll(freqs, -freqcenter, axis=0)
    lastresult = None

    for i in x:
        result = stride_windows(i, NFFT, noverlap, axis=0)
        result = detrend(result, detrend_func, axis=0)
        result = fft(result, n=pad_to, axis=0)[:numFreqs, :]  # pylint: disable=invalid-sequence-index
        if not np.iterable(window):
            window = window(np.ones(NFFT, i.dtype))
        if len(window) != NFFT:
            raise ValueError(
                "The window length must match the data's first dimension")

        if mode == 'psd':
            result = np.conj(result) * result
        elif mode == 'magnitude':
            result = np.abs(result) / np.abs(window).sum()
        elif mode == 'angle' or mode == 'phase':
            # we unwrap the phase later to handle the onesided vs. twosided case
            result = np.angle(result)
        elif mode == 'complex':
            result /= np.abs(window).sum()

        if mode == 'psd':
            # Also include scaling factors for one-sided densities and dividing by
            # the sampling frequency, if desired. Scale everything, except the DC
            # component and the NFFT/2 component:

            # if we have a even number of frequencies, don't scale NFFT/2
            if not NFFT % 2:
                slc = slice(1, -1, None)
            # if we have an odd number, just don't scale DC
            else:
                slc = slice(1, None, None)

            result[slc] *= scaling_factor

            # MATLAB divides by the sampling frequency so that density function
            # has units of dB/Hz and can be integrated by the plotted frequency
            # values. Perform the same scaling here.
            if scale_by_freq:
                result /= Fs
                # Scale the spectrum by the norm of the window to compensate for
                # windowing loss; see Bendat & Piersol Sec 11.5.2.
                result /= (np.abs(window)**2).sum()
            else:
                # In this case, preserve power in the segment, not amplitude
                result /= np.abs(window).sum()**2

        # center the frequency range at zero
        result = np.roll(result, -freqcenter, axis=0)

        # we unwrap the phase here to handle the onesided vs. twosided case
        if mode == 'phase':
            result = np.unwrap(result, axis=0)
        result = np.apply_along_axis(np.real, 1, result)

        if lastresult is None:
            lastresult = result
        else:
            lastresult = np.hstack((lastresult, result))

    t_x = lastresult.shape[0] * lastresult.shape[1]
    t = np.arange(NFFT/2, t_x - NFFT/2 + 1, NFFT - noverlap)/Fs
    return lastresult, freqs, t


def specgram(x, NFFT=None, Fs=None, Fc=None, detrend=None,
             window=None, noverlap=None,
             xextent=None, pad_to=None,
             scale_by_freq=None, mode=None, scale=None,
             **kwargs):
    if NFFT is None:
        NFFT = 256  # same default as in mlab.specgram()
    if Fc is None:
        Fc = 0  # same default as in mlab._spectral_helper()
    if noverlap is None:
        noverlap = 128  # same default as in mlab.specgram()
    if Fs is None:
        Fs = 2  # same default as in mlab._spectral_helper()

    if mode == 'complex':
        raise ValueError('Cannot plot a complex specgram')

    if scale is None or scale == 'default':
        if mode in ['angle', 'phase']:
            scale = 'linear'
        else:
            scale = 'dB'
    elif mode in ['angle', 'phase'] and scale == 'dB':
        raise ValueError('Cannot use dB scale with angle or phase mode')

    spec, freqs, t = spectral_helper(x=x, NFFT=NFFT, Fs=Fs,
                                     detrend_func=detrend, window=window,
                                     noverlap=noverlap, pad_to=pad_to,
                                     scale_by_freq=scale_by_freq,
                                     mode=mode)

    if scale == 'linear':
        Z = spec
    elif scale == 'dB':
        if mode is None or mode == 'default' or mode == 'psd':
            Z = 10. * np.log10(spec)
        else:
            Z = 20. * np.log10(spec)
    else:
        raise ValueError(f'Unknown scale {scale!r}')

    Z = np.flipud(Z)

    if xextent is None:
        # padding is needed for first and last segment:
        pad_xextent = (NFFT-noverlap) / Fs / 2
        xextent = np.min(t) - pad_xextent, np.max(t) + pad_xextent
    xmin, xmax = xextent
    freqs += Fc
    extent = xmin, xmax, freqs[0], freqs[-1]

    if 'origin' in kwargs:
        raise TypeError('specgram() got an unexpected keyword argument '
                        "'origin'")

    return Z, extent


def get_reader(filename):

    def gzip_reader(x):
        return gzip.open(x, 'rb')

    def zst_reader(x):
        return zstandard.ZstdDecompressor().stream_reader(open(x, 'rb'))

    def default_reader(x):
        return open(x, 'rb')

    if filename.endswith('.gz'):
        return gzip_reader
    if filename.endswith('.zst'):
        return zst_reader

    return default_reader


def read_recording(filename, sample_rate):
    # TODO: assume int16.int16 (parse from sigmf).
    dtype = np.dtype([('i', '<i2'), ('q', '<i2')])
    dtype_size = 4

    reader = get_reader(filename)
    with reader(filename) as infile:
        while True:
            sample_buffer = infile.read(sample_rate * dtype_size)
            buffered_samples = int(len(sample_buffer) / dtype_size)
            if buffered_samples == 0:
                break
            x1d = np.frombuffer(sample_buffer, dtype=dtype,
                                count=buffered_samples)
            yield x1d['i'] + np.csingle(1j) * x1d['q']


def plot_spectrogram(x, spectrogram_filename, nfft, fs, fc, cmap):
    fig = plt.figure()
    fig.set_size_inches(11, 8)
    axes = fig.add_subplot(111)
    axes.set_xlabel('time (s)')
    axes.set_ylabel('freq (Hz)')
    # overlap must be 0, for maximum detail.
    Z, extent = specgram(
        x, NFFT=nfft, Fs=fs, cmap=cmap, Fc=fc, noverlap=0)
    im = axes.imshow(Z, cmap=cmap, extent=extent, origin='upper')
    axes.axis('auto')
    plt.sci(im)
    plt.savefig(spectrogram_filename)
    # must call this in specific order to avoid pyplot leak
    axes.images.remove(im)
    fig.clear()
    plt.close()
    plt.cla()
    plt.clf()


def process_recording(args, recording):
    print(f'processing {recording}')
    freq_center, sample_rate = parse_filename(recording)
    samples = read_recording(recording, sample_rate)
    plot_spectrogram(
        samples,
        replace_ext(recording, 'jpg'),
        args.nfft,
        sample_rate,
        freq_center,
        cmap=args.cmap)


def main():
    parser = argparse.ArgumentParser(
        description='draw spectrogram from recording')
    parser.add_argument('recording', default='', type=str,
                        help='filename of recording, or directory')
    parser.add_argument('--nfft', default=int(65536), type=int,
                        help='number of FFT points')
    parser.add_argument('--cmap', default='twilight_r', type=str,
                        help='pyplot colormap (see https://matplotlib.org/stable/tutorials/colors/colormaps.html)')
    args = parser.parse_args()

    if os.path.isdir(args.recording):
        recordings = get_nondot_files(args.recording)
    else:
        recordings = [args.recording]
    for recording in recordings:
        process_recording(args, recording)


if __name__ == '__main__':
    main()
