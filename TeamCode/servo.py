import serial
import time

ser = serial.Serial('/dev/tty.usbserial-1120', 115200, timeout=0.5)

while True:
    ser.write(b'set_servo_position(16, 0)\r')
    time.sleep(10)
    ser.write(b'set_servo_position(16, -0.9)\r')
    time.sleep(10)
    # ser.write(b'set_servo_position(16, 0.9)\r')
    # time.sleep(1)

ser.close()