from gpiozero import Servo
from time import sleep


servo = Servo(13)

def ServoDrill():
    servo.max()
    sleep(15.0)
    servo.mid()
    sleep(2.0)
    servo.min()
    sleep(12.0)
    
    
