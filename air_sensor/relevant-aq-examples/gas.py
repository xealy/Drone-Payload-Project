#!/usr/bin/env python3

import logging
import time

from enviroplus import gas

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S")

logging.info("""gas.py - Print readings from the MICS6814 Gas sensor.

Press Ctrl+C to exit!

""")

try:
    while True:
        readings = gas.read_all()
        logging.info(readings)
        time.sleep(1.0)
except KeyboardInterrupt:
    pass
