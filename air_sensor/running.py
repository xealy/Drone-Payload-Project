from gpiozero import AngularServo
import RPi.GPIO as GPIO
import time
import pigpio


GPIO.setmode(GPIO.BCM)
# pin on board 
mosfetpin = 24

GPIO.setup(mosfetpin, GPIO.OUT)
pwm24 = GPIO.PWM(mosfetpin, 1)


pwm = pigpio.pi()
pwm.set_mode(8, pigpio.OUTPUT)
pwm.set_PWM_frequency(8, 50)

GPIO.setup(14, GPIO.IN, pull_up_down = GPIO.PUD_DOWN)


def ServoDrill(total_time, direction):
    GPIO.setup(mosfetpin, GPIO.OUT)
    pwm24.start(100)
    liftpin = 1 # dummy value
    drill = AngularServo(liftpin, min_pulse_width=0.001, max_pulse_width=0.002) # what is lift pin ?????
    if direction == 'stop':
        drill.angle = 0;
    else :
        while (total_time > 0):
            if direction == 'up' :
                drill.angle = 90;
            elif direction == 'down':
                if(GPIO.input(14) == GPIO.HIGH):
                    drill.angle = 90
                    time.sleep(1.8) # will need to change depending on how far down it needs to go 
                    break
                drill.angle = -90
            
            print(total_time)    
            time.sleep(0.5)
            total_time -= 1
            
    pwm24.stop()
    

# GPIO.cleanup()

