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
# ser = serial.Serial('/dev/tty.usbmodem58CF0752231', )
#ser = serial.Serial('/dev/cu.BittleE9_SSP', 921600)
print('Creating serial port')
ser = serial.Serial('/dev/rfcomm0', 115200)

#while not keyboard.is_pressed('space'):
#while not any_key_pressed:

print('Sending Data')
for i in range(10):
    u = np.random.uniform()
    if u < 0.4:
        ser.write("kbk".encode('utf-8'))
        time.sleep(2)
    elif u > 0.6:
        ser.write('kwkF'.encode('utf-8'))
        time.sleep(2)
    else:
        ser.write('kbalance'.encode('utf-8'))
        time.sleep(2)

print('Send Complete')
ser.write("kbalance".encode('utf-8'))

ser.close()
