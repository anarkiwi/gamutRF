import time
import sys
from argparse import ArgumentParser, BooleanOptionalAction

try:
    from gnuradio import iqtlabs  # pytype: disable=import-error
    from gnuradio import gr  # pytype: disable=import-error
except ModuleNotFoundError as err:  # pragma: no cover
    print(
        "Run from outside a supported environment, please run via Docker (https://github.com/IQTLabs/gamutRF#readme): %s"
        % err
    )
    sys.exit(1)


from gamutrf.grfft import grfft


def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--nfft",
        dest="nfft",
        type=int,
        default=2048,
        help="FFTI size [default=%(default)r]",
    )
    parser.add_argument(
        "--fft_batch_size",
        dest="fft_batch_size",
        type=int,
        default=256,
        help="offload FFT batch size",
    )
    parser.add_argument(
        "--vkfft",
        dest="vkfft",
        default=True,
        action=BooleanOptionalAction,
        help="use VkFFT (ignored if wavelearner available)",
    )
    return parser


def main():
    options = argument_parser().parse_args()

    wavelearner = None
    try:
        import wavelearner as wavelearner_lib  # pytype: disable=import-error

        wavelearner = wavelearner_lib
        print("using wavelearner")
    except ModuleNotFoundError:
        print("wavelearner not available")

    fft_args = {
        "iqtlabs": iqtlabs,
        "wavelearner": wavelearner,
    }
    fft_args.update(
        {k: getattr(options, k) for k in dir(options) if not k.startswith("_")}
    )
    tb = grfft(**fft_args)
    tb.start()
    while True:
        time.sleep(1)
    tb.stop()
    tb.wait()
