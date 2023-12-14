#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

try:
    from gnuradio import blocks  # pytype: disable=import-error
    from gnuradio import fft  # pytype: disable=import-error
    from gnuradio import gr  # pytype: disable=import-error
    from gnuradio import zeromq  # pytype: disable=import-error
    from gnuradio.fft import window  # pytype: disable=import-error
except ModuleNotFoundError as err:  # pragma: no cover
    print(
        "Run from outside a supported environment, please run via Docker (https://github.com/IQTLabs/gamutRF#readme): %s"
        % err
    )
    sys.exit(1)


class grfft(gr.top_block):
    def __init__(
        self,
        fft_batch_size=256,
        iqtlabs=None,
        nfft=1024,
        vkfft=False,
        wavelearner=None,
    ):
        gr.top_block.__init__(self, "fft", catch_exceptions=True)

        self.nfft = nfft
        self.wavelearner = wavelearner
        self.iqtlabs = iqtlabs

        out_zmq_addr = "tcp://0.0.0.0:11001"
        in_zmq_addr = "tcp://0.0.0.0:11002"
        fft_batch_size, fft_blocks = self.get_offload_fft_blocks(
            vkfft, fft_batch_size, nfft
        )

        fft_blocks = (
            [
                zeromq.sub_source(
                    gr.sizeof_gr_complex,
                    fft_batch_size * nfft,
                    in_zmq_addr,
                    timeout=100,
                    pass_tags=True,
                    hwm=65536,
                    key="",
                    bind=True,
                )
            ]
            + fft_blocks
            + [
                zeromq.pub_sink(
                    gr.sizeof_gr_complex,
                    nfft,
                    out_zmq_addr,
                    timeout=100,
                    pass_tags=True,
                    hwm=65536,
                    key="",
                )
            ]
        )
        self.connect_blocks(fft_blocks[0], fft_blocks[1:])

    def connect_blocks(self, source, other_blocks, last_block_port=0):
        last_block = source
        for block in other_blocks:
            self.connect((last_block, last_block_port), (block, 0))
            last_block = block

    def get_window(self, nfft):
        return window.hann(nfft)

    def get_offload_fft_blocks(
        self,
        vkfft,
        fft_batch_size,
        nfft,
    ):
        fft_block = None
        fft_roll = False
        if self.wavelearner:
            fft_block = self.wavelearner.fft(int(fft_batch_size * nfft), (nfft), True)
            fft_roll = True
        elif vkfft:
            fft_block = self.iqtlabs.vkfft(int(fft_batch_size * nfft), nfft, True)
        else:
            fft_batch_size = 1
            fft_blocks = [
                fft.fft_vcc(nfft, True, self.get_window(nfft), True, 1),
            ]
            return fft_batch_size, fft_blocks

        fft_blocks = [
            blocks.multiply_const_vff(
                [val for val in self.get_window(nfft) for _ in range(2)]
                * fft_batch_size
            ),
            fft_block,
            blocks.vector_to_stream(gr.sizeof_gr_complex * nfft, fft_batch_size),
        ]
        if fft_roll:
            fft_blocks.append(self.iqtlabs.vector_roll(nfft))
        return fft_batch_size, fft_blocks
