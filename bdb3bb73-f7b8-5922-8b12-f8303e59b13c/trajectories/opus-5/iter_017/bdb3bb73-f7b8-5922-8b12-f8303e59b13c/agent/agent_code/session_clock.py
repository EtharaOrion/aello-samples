import time, os
T0 = 1787034330  # session start (unix sec), captured at first tool call
def elapsed(): return time.time() - T0
def remaining(): return 3600 - elapsed()
if __name__ == "__main__":
    print(f"elapsed={elapsed():.0f}s remaining={remaining():.0f}s")
