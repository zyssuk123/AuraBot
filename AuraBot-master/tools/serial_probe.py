import os
import time

import serial
from dotenv import load_dotenv


def main():
    load_dotenv()
    port = os.getenv("ARDUINO_PORT", "COM5")
    baud = int(os.getenv("BAUD_RATE", "9600"))
    duration = float(os.getenv("SERIAL_PROBE_SECONDS", "5"))

    print(f"Opening {port} at {baud} baud...")
    ser = serial.Serial(port, baud, timeout=0.2, write_timeout=0.2)
    try:
        time.sleep(2)
        ser.reset_input_buffer()
        print("Connected. Waiting for Arduino lines...")
        deadline = time.time() + duration
        count = 0
        while time.time() < deadline:
            raw = ser.readline()
            if not raw:
                continue
            text = raw.decode("ascii", errors="backslashreplace").strip()
            print("RX:", text, "| bytes:", raw.hex(" "))
            count += 1
        print(f"Done. Lines received: {count}")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
