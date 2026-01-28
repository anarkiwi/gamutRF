#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import logging
import sys
import threading
import time
import pmt
import zmq

NO_SIGNAL = "No signal"

try:
    from gnuradio import gr  # pytype: disable=import-error
    from gamutrf.mqtt_reporter import MQTTReporter
except ModuleNotFoundError as err:  # pragma: no cover
    print(
        "Run from outside a supported environment, please run via Docker (https://github.com/IQTLabs/gamutRF#readme): %s"
        % err
    )
    sys.exit(1)


class inferenceoutput(gr.basic_block):
    def __init__(
        self,
        name,
        zmq_addr,
        mqtt_server,
        compass,
        gps_server,
        use_external_gps,
        use_external_heading,
        external_gps_server,
        external_gps_server_port,
        log_path,
        nest,
    ):
        gr.basic_block.__init__(
            self,
            name="inferenceoutput",
            in_sig=None,
            out_sig=None,
        )
        self.serialno = 0
        self.start_time = time.time()
        self.zmq_context = None
        self.zmq_pub = None
        if zmq_addr:
            self.zmq_context = zmq.Context()
            self.zmq_pub = self.zmq_context.socket(zmq.PUB)
            self.zmq_pub.setsockopt(zmq.SNDHWM, 100)
            self.zmq_pub.setsockopt(zmq.SNDBUF, 65536)
            self.zmq_pub.bind(zmq_addr)
        self.mqtt_reporter = None
        self.log_path = log_path
        if mqtt_server:
            self.mqtt_reporter = MQTTReporter(
                name=name,
                mqtt_server=mqtt_server,
                gps_server=gps_server,
                compass=compass,
                use_external_gps=use_external_gps,
                use_external_heading=use_external_heading,
                external_gps_server=external_gps_server,
                external_gps_server_port=external_gps_server_port,
                nest=nest,
            )
        self.message_port_register_in(pmt.intern("inference"))
        self.set_msg_handler(pmt.intern("inference"), self.receive_pdu)
        self.nest_thread = None
        self.running = True
        if nest:
            self.nest_thread = threading.Thread(target=self.nest_hb)
            self.nest_thread.start()

    def nest_hb(self):
        while self.running:
            self.publish_pdu({"metadata": {"ts": time.time()}}, event_type="heartbeat")
            time.sleep(5)

    def stop(self):
        self.running = False
        if self.zmq_pub is not None:
            self.zmq_pub.close()
        if self.nest_thread:
            self.nest_thread.join()

    def receive_pdu(self, pdu):
        item = json.loads(bytes(pmt.to_python(pmt.cdr(pdu))).decode("utf8"))
        try:
            predictions = set(item["predictions"].keys())
            if NO_SIGNAL in predictions:
                return
        except KeyError:
            pass
        self.publish_pdu(item)

    def publish_pdu(self, item, event_type="detect"):
        self.serialno += 1
        logging.info("inference output %u: %s", self.serialno, item)
        if self.zmq_pub is not None and item:
            self.zmq_pub.send_string(json.dumps(item), flags=zmq.NOBLOCK)
        if self.mqtt_reporter is not None:
            self.mqtt_reporter.publish("gamutrf/inference", item, event_type=event_type)
            self.mqtt_reporter.log(self.log_path, "inference", self.start_time, item)
