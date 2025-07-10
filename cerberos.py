import time
import serial
import numpy as np
# import keyboard
# from pynput import keyboard

any_key_pressed = False
#def on_press(key):
#    any_key_pressed = True
#with keyboard.Listener(on_press=on_press) as listener:
#    listener.join()

# Open serial port (adjust 'COM3' or '/dev/ttyUSB0' and baudrate as needed)
# ser = serial.Serial('/dev/tty.usbmodem58CF0752231', 115200)
ser = serial.Serial('/dev/cu.BittleE9_SSP', 921600)

#while not keyboard.is_pressed('space'):
#while not any_key_pressed:

start = time.time()
state = "idle"
while time.time() - start < 20:
    alpha = np.sin(time.time())
    if alpha > 0.1 and state != "walk":
        state = "walk"
        ser.write("kwkF".encode('utf-8'))
        print(state)
    elif alpha < -0.5 and state != "sit":
        state = "sit"
        ser.write("ksit".encode('utf-8'))
        print(state)
    time.sleep(.1)

ser.write("ksit".encode('utf-8'))
#ser.write(".".encode('utf-8'))

ser.close()
