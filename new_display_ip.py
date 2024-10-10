# working IP display, with current codes rearranged in functions
# Emma

import logging
import st7735
from fonts.ttf import RobotoMedium as UserFont
from PIL import Image, ImageDraw, ImageFont
import socket

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connect to a host to retrieve the IP address
        s.connect(('8.8.8.8', 80))
        ipaddr = s.getsockname()[0]
    except Exception:
        # Catch undesired IP address results
        ipaddr = '127.0.0.1'
    finally:
        s.close()
    return ipaddr

def setup_display():
    # Create LCD class instance.
    disp = st7735.ST7735(
        port=0,
        cs=1,
        dc=9,
        backlight=12,
        rotation=270,
        spi_speed_hz=10000000
    )
    return disp

def display_ip(disp, message):
    # Width and height to calculate text position.
    WIDTH = disp.width
    HEIGHT = disp.height

    # New canvas to draw on.
    img = Image.new("RGB", (WIDTH, HEIGHT), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Text settings.
    font_size = 25
    font = ImageFont.truetype(UserFont, font_size)
    text_colour = (255, 255, 255)
    back_colour = (0, 170, 170)

    x1, y1, x2, y2 = font.getbbox(message)
    size_x = x2 - x1
    size_y = y2 - y1

    # Calculate text position
    x = (WIDTH - size_x) / 2
    y = (HEIGHT / 2) - (size_y / 2)

    # Draw background rectangle and write text.
    draw.rectangle((0, 0, 160, 80), back_colour)
    draw.text((x, y), message, font=font, fill=text_colour)
    disp.display(img)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    logging.info("""This code displays the IP address onto the LCD. Press Ctrl+C to exit!""")

    # Initialize display
    disp = setup_display()
    disp.begin()

    # Get IP address and display it
    message = get_ip()
    display_ip(disp, message)