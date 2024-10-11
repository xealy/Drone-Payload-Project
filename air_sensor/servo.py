from gpiozero import Servo
from time import sleep
 
servo = Servo(4)
 
while True:
    servo.mid()
    print("mid")
    sleep(0.5)
    servo.min()
    print("min")
    sleep(1)
    servo.mid()
    print("mid")
    sleep(0.5)
    servo.max()
    print("max")
    sleep(1)
    break 
servo.mid() 
sleep(0.5)
