from gpiozero import Servo
from time import sleep

# servo = Servo(13)

# def ServoDrill():
#     servo.max()
#     sleep(15.0)
#     servo.mid()
#     sleep(2.0)
#     servo.min()
#     sleep(12.0)
    
class ServoClass(object):
    def __init__(self):
        self.servo = Servo(13)
    
    def start_servo(self):
        self.servo.max()
        sleep(15.0)
        self.servo.mid()
        sleep(2.0)
        self.servo.min()
        sleep(12.0)

