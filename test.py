import serial
import time

with serial.Serial("COM3", 115200, timeout=100, dsrdtr=None) as ser:

    ser.setRTS(False)
    ser.setDTR(False)

    data1 = bytes([0, 50])
    data2 = bytes([1, 180])

    ser.write(data1)
    ser.write(data2)
