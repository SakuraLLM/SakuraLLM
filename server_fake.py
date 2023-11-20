import time
import sys

while True:
    print("Hello world", flush=True)
    print("Bye", file=sys.stderr)
    time.sleep(1)
