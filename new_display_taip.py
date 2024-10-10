# working TAIP display, with current codes rearranged in functions
# but can't be stopped by pressing 'q' (see fun run_pipeline)
# Emma

import sys
import numpy as np
import cv2
from PIL import Image
import st7735

import depthai as dai
import logging

def setup_display():
    """Create and initialize the ST7735 LCD display."""
    try:
        disp = st7735.ST7735(
            port=0,
            cs=1,
            dc=9,
            backlight=12,
            rotation=270,
            spi_speed_hz=10000000
        )
        disp.begin()
        logging.info("Display initialized successfully.")
        return disp
    except Exception as e:
        logging.error(f"Error initializing display: {e}")
        raise

def create_pipeline():
    """Create and configure the DepthAI pipeline for the color camera."""
    try:
        pipeline = dai.Pipeline()

        # Define source and output
        camRgb = pipeline.create(dai.node.ColorCamera)
        xoutRgb = pipeline.create(dai.node.XLinkOut)

        xoutRgb.setStreamName("rgb")

        # Properties (use the actual WIDTH and HEIGHT for your display)
        WIDTH = 160  # Adjust accordingly
        HEIGHT = 80  # Adjust accordingly
        camRgb.setPreviewSize(WIDTH, HEIGHT)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

        # Linking
        camRgb.preview.link(xoutRgb.input)

        logging.info("Pipeline created successfully.")
        return pipeline
    except Exception as e:
        logging.error(f"Error creating pipeline: {e}")
        raise

def display_frame(disp, frame):
    """Display the given frame on the LCD display."""
    try:
        # Convert the frame from OpenCV format (BGR) to RGB and then to PIL Image
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img_rgb)

        # Resize the image to fit the display
        im_pil = im_pil.resize((disp.width, disp.height))

        # Display image on LCD
        disp.display(im_pil)
    except Exception as e:
        logging.error(f"Error displaying frame: {e}")


def run_pipeline(disp):
    """Run the DepthAI pipeline and display frames on the LCD."""
    try:
        pipeline = create_pipeline()

        # Connect to the device and start the pipeline
        with dai.Device(pipeline) as device:
            qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

            while True:
                inRgb = qRgb.get()
                frame = inRgb.getCvFrame()

                # Display the frame on the LCD
                display_frame(disp, frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) == ord('q'):
                    break

        cv2.destroyAllWindows()
        logging.info("Pipeline run successfully.")
    except Exception as e:
        logging.error(f"Error running pipeline: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    logging.info("Starting the display and DepthAI pipeline...")

    # Initialize the display
    disp = setup_display()

    # Run the DepthAI pipeline and display frames
    run_pipeline(disp)

