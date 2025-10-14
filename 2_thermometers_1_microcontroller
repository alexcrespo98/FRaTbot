import serial
import csv
import time
import re
from datetime import datetime
import threading

# --- CONFIG ---
PORT = "COM19"
BAUD = 9600
DURATION = 10  # seconds, fixed
NUM_RUNS = 4
SAMPLE_RATE = 10  # Hz (samples per second)
SAMPLES_PER_RUN = DURATION * SAMPLE_RATE

# --- CONNECT ---
print("Connecting to Arduino...")
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)
print("Connected.\n")

def relay_on():
    ser.write(b'RELAY ON\n')

def relay_off():
    ser.write(b'RELAY OFF\n')

def relay_oscillation_thread(half_period, stop_event):
    relay_state = False
    relay_off()
    while not stop_event.is_set():
        if relay_state:
            relay_off()
        else:
            relay_on()
        relay_state = not relay_state
        # Wait for half period or until stop event is set
        stop_event.wait(timeout=half_period)
    relay_off()

# Get flow rate and suggest filename
flow = input("Enter flow rate label for this test (e.g. 5GPM): ").strip()
if not flow:
    flow = "test"
default_filename = f"{flow}_10s_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
filename = input(f"Enter CSV filename to save (default: {default_filename}): ").strip()
if not filename:
    filename = default_filename

print(f"Data will be saved to: {filename}")

# --- MAIN LOOP ---
all_runs = []
headers = []

osc_period = None  # Will be set by user

for run in range(1, NUM_RUNS + 1):
    # Oscillation period prompt (after first run, ask after each recording)
    while True:
        if osc_period is None:
            osc_str = input(f"Enter relay oscillation period for run {run} (seconds per full on/off cycle, e.g. 1): ").strip()
        else:
            osc_str = input(f"Enter NEW relay oscillation period for run {run} (last was {osc_period}s): ").strip()
        try:
            period = float(osc_str)
            if period > 0:
                osc_period = period
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")

    half_period = osc_period / 2
    stop_event = threading.Event()
    osc_thread = threading.Thread(target=relay_oscillation_thread, args=(half_period, stop_event))
    osc_thread.start()

    print(f"\nRelay is now oscillating at {osc_period}s per cycle ({half_period}s ON, {half_period}s OFF).")
    input(f"Press Enter when ready to START recording run {run} for {flow} ({DURATION}s)...")

    data = []

    start_time = time.time()
    for sample_count in range(SAMPLES_PER_RUN):
        now = time.time()
        elapsed = now - start_time

        # Read line from serial (non-blocking)
        ser.timeout = 0.05
        line = ser.readline().decode(errors='ignore').strip()
        match = re.search(r"T1=([\d.]+)C,\s*T2=([\d.]+)C", line)
        if match:
            t1, t2 = match.groups()
        else:
            t1, t2 = "", ""
        timestamp = round(elapsed, 2)
        data.append([timestamp, t1, t2])

        # Wait for next sample time
        target_next = start_time + (sample_count + 1) * (1 / SAMPLE_RATE)
        sleep_time = target_next - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)

    # End relay oscillation
    stop_event.set()
    osc_thread.join()
    relay_off()
    print(f"Recorded {len(data)} samples for run {run}\n")

    all_runs.append(data)
    headers.extend([f"timestamp_run{run}", f"T1_run{run}", f"T2_run{run}"])

ser.close()

# --- COMBINE AND SAVE ---
# Merge all runs by columns
merged = []
for i in range(SAMPLES_PER_RUN):
    row = []
    for run in all_runs:
        row.extend(run[i])
    merged.append(row)

with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(merged)

print(f"✅ Saved all {NUM_RUNS} runs ({len(merged)} rows) to {filename}")
