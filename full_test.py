#!/usr/bin/env python3
# full_test.py
# Column-oriented output: 1 metadata row + N sample rows (e.g. 501) and one column per thermometer per run.
# This revision:
# - Does NOT wait for "Ready." and does NOT print ESP boot logs at the prompt.
# - Flushes/consumes startup boot chatter immediately after opening the serial port.
# - Starts the SerialReader so live temps are visible while you wait.
# - When starting a run we flush any backlog BEFORE sending start_stream (so we do not discard fresh stream samples).
# - Shows "Oscillating at X Hz. Press Enter when temperatures stabilize" before the live display.
# - On Enter, prints "Recording now, please wait N seconds...", records for exactly sec_data seconds (wall time),
#   then prints a clear multi-line summary and leaves a few blank lines for readability.
# - Improves robustness around discarding an initial bad reading by invalidating the display briefly after start_stream.
#
# Edit COM_PORT if needed and run: python full_test.py
# Requires: pyserial

import serial
import threading
import time
import csv
import os
import sys
from datetime import datetime

COM_PORT = "COM20"
BAUD = 115200
SERIAL_TIMEOUT = 0.2

# Validation thresholds (tune if needed)
MIN_VALID_TEMP = -50.0   # Celsius
MAX_VALID_TEMP = 150.0   # Celsius
REQUIRED_VALID_STREAK = 2  # number of consecutive valid samples to treat as "validated"

def read_input(prompt, default=None, cast=str):
    try:
        inp = input(prompt)
    except EOFError:
        inp = ''
    if inp == '':
        return cast(default) if default is not None else None
    return cast(inp)


class SerialReader(threading.Thread):
    def __init__(self, ser, use_ivan=False, display_interval_ms=100, show_acks=False):
        super().__init__(daemon=True)
        self.ser = ser
        self.use_ivan = use_ivan
        self.show_acks = show_acks

        self.running = threading.Event()
        self.recording = threading.Event()
        self.lock = threading.Lock()

        # latest_raw_values: always updated when a numeric sample is parsed
        # latest_valid_values: updated only after a required valid streak
        self.latest_raw_values = []
        self.latest_valid_values = []
        self.recorded = []

        self.running.set()
        self.last_display_time = 0.0
        self.display_interval_ms = max(1, int(display_interval_ms))
        self._last_printed_len = 0

        self.ignore_until = 0.0
        self.valid_streak = 0

    def ignore_for(self, seconds):
        with self.lock:
            self.ignore_until = time.time() + float(seconds)
            self.valid_streak = 0  # reset streak

    def invalidate(self):
        with self.lock:
            self.valid_streak = 0
            self.latest_valid_values = []
            # keep raw values so the live read shows something immediately

    def start_recording(self):
        with self.lock:
            self.recorded = []
        self.recording.set()

    def stop_recording(self):
        self.recording.clear()

    def get_recorded_copy(self):
        with self.lock:
            return list(self.recorded)

    def _print_live(self):
        # Prefer validated values if available; otherwise show raw (tagged)
        with self.lock:
            valid = self.latest_valid_values[:]
            raw = self.latest_raw_values[:]

        if valid:
            # show validated reading
            if self.use_ivan and len(valid) >= 3:
                s = f"Prox: {valid[0]:6.3f} C  Dist: {valid[1]:6.3f} C  Ivan: {valid[2]:6.3f} C"
            else:
                s = f"Prox: {valid[0]:6.3f} C  Dist: {valid[1]:6.3f} C"
        elif raw:
            # show raw reading but mark it
            if self.use_ivan and len(raw) >= 3:
                s = f"RAW Prox: {raw[0]:6.3f} C  Dist: {raw[1]:6.3f} C  Ivan: {raw[2]:6.3f} C"
            else:
                s = f"RAW Prox: {raw[0]:6.3f} C  Dist: {raw[1]:6.3f} C"
        else:
            return

        pad = max(0, self._last_printed_len - len(s))
        try:
            sys.stdout.write("\r" + s + " " * pad)
            sys.stdout.flush()
        except UnicodeEncodeError:
            sys.stdout.write("\rTemps available (unicode error)      ")
            sys.stdout.flush()
        self._last_printed_len = len(s)

    def run(self):
        while self.running.is_set():
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            except Exception:
                line = ''
            if line:
                # ignore ACKs unless explicitly requested
                if line.startswith("ACK:") and not self.show_acks:
                    continue

                parts = [p.strip() for p in line.split(',') if p.strip() != '']
                numeric_parts = []
                for p in parts:
                    try:
                        numeric_parts.append(float(p))
                    except:
                        numeric_parts = None
                        break

                if numeric_parts is not None and len(numeric_parts) >= 2:
                    # always update raw values (unless we're in ignore window)
                    if time.time() < self.ignore_until:
                        continue

                    with self.lock:
                        self.latest_raw_values = numeric_parts
                        # If recording, append raw numeric sample immediately (so recording captures all samples)
                        if self.recording.is_set():
                            self.recorded.append(list(numeric_parts))

                    # do validation on the parsed numeric parts for display only
                    valid = True
                    needed_fields = 3 if self.use_ivan else 2
                    for v in numeric_parts[:needed_fields]:
                        if v is None or v != v:
                            valid = False
                            break
                        if v < MIN_VALID_TEMP or v > MAX_VALID_TEMP:
                            valid = False
                            break

                    with self.lock:
                        if valid:
                            self.valid_streak += 1
                        else:
                            self.valid_streak = 0

                        if self.valid_streak >= REQUIRED_VALID_STREAK:
                            # accept as validated reading (for display)
                            self.latest_valid_values = numeric_parts

                    now_ms = time.time() * 1000.0
                    if now_ms - self.last_display_time >= self.display_interval_ms:
                        self.last_display_time = now_ms
                        # always print something (raw or validated) so user sees activity
                        self._print_live()
                else:
                    # non-numeric lines (boot logs, ACKs if allowed)
                    # Only print these if the user asked to see ACKs/status
                    if self.show_acks:
                        try:
                            sys.stdout.write("\n" + line + "\n")
                            sys.stdout.flush()
                        except UnicodeEncodeError:
                            sys.stdout.write("\n<non-ASCII serial line>\n")
                            sys.stdout.flush()
                    # after printing a non-temp line, reprint live info (raw or validated)
                    self._print_live()
            else:
                time.sleep(0.005)

    def stop(self):
        self.running.clear()


def wait_for_enter_with_timer(prompt_text):
    """
    Wait for the user to press Enter. While waiting, display elapsed time every second.
    Uses msvcrt on Windows and select on POSIX so the serial reader thread can continue updating the live line.
    """
    start = time.time()
    last_print = 0.0
    if os.name == 'nt':
        import msvcrt
        sys.stdout.write("\n" + prompt_text + "\n")
        sys.stdout.write("Elapsed: 0s")
        sys.stdout.flush()
        while True:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch == '\r' or ch == '\n':
                    break
            now = time.time()
            if now - last_print >= 1.0:
                elapsed = int(now - start)
                try:
                    sys.stdout.write("\rElapsed: {}s ".format(elapsed))
                    sys.stdout.flush()
                except UnicodeEncodeError:
                    pass
                last_print = now
            time.sleep(0.05)
        sys.stdout.write("\n")
        sys.stdout.flush()
    else:
        import select
        sys.stdout.write("\n" + prompt_text + "\n")
        sys.stdout.write("Elapsed: 0s")
        sys.stdout.flush()
        while True:
            rlist, _, _ = select.select([sys.stdin], [], [], 0.5)
            if rlist:
                _ = sys.stdin.readline()
                break
            now = time.time()
            if now - last_print >= 1.0:
                elapsed = int(now - start)
                try:
                    sys.stdout.write("\rElapsed: {}s ".format(elapsed))
                    sys.stdout.flush()
                except UnicodeEncodeError:
                    pass
                last_print = now
        sys.stdout.write("\n")
        sys.stdout.flush()


def flush_input_buffer_safe(ser):
    # Try modern reset_input_buffer(); fallback to flushInput for older pyserial
    try:
        ser.reset_input_buffer()
    except Exception:
        try:
            ser.flushInput()
        except Exception:
            pass


def format_filename(prefix="flowdata"):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return f"{prefix}_{ts}.csv"


def main():
    global COM_PORT
    print("ESP32 dual-therm controller (column layout: one column per thermometer per run)")
    COM_PORT = read_input(f"Serial port [{COM_PORT}]: ", COM_PORT, str)

    try:
        ser = serial.Serial(COM_PORT, BAUD, timeout=SERIAL_TIMEOUT)
    except Exception as e:
        print("Failed to open serial port:", e)
        return

    # Give the port a short moment for the ESP to reset and produce any boot messages,
    # then flush them so they don't appear at prompts.
    time.sleep(0.3)
    flush_input_buffer_safe(ser)

    # Start the reader early so live temps are visible during setup and while you wait.
    reader = SerialReader(ser, use_ivan=False, display_interval_ms=100, show_acks=False)
    reader.start()

    print("Connected to ESP32 on", COM_PORT)

    num_freqs = int(read_input("Number of oscillation frequencies [4]: ", "4", int))
    freq_str = read_input("Oscillation frequencies (Hz, comma separated) [1,1.5,2,2.5]: ", "1,1.5,2,2.5", str)
    freqs = [float(f.strip()) for f in freq_str.split(',') if f.strip()]

    num_flows = int(read_input("Number of flow readings [4]: ", "4", int))
    use_ivan = int(read_input("Use optional Ivan thermometer? 0=no,1=yes [0]: ", "0", int)) != 0

    # Inform reader of Ivan selection (so display will include it)
    reader.use_ivan = use_ivan

    sec_data = float(read_input("Seconds of data per frequency [5]: ", "5", float))
    interval_ms = int(read_input("Temperature reading interval (ms) [10]: ", "10", int))
    display_rate_ms = int(read_input("Live display update interval (ms) [100]: ", "100", int))
    show_acks = int(read_input("Show ACK/status lines on console? 0=no,1=yes [0]: ", "0", int)) != 0

    # Apply dynamic settings to reader
    reader.display_interval_ms = max(1, int(display_rate_ms))
    reader.show_acks = show_acks

    # expected samples: preserve +1 behavior so 5s @10ms => 501
    expected_samples = int(round(sec_data * 1000.0 / interval_ms)) + 1 if interval_ms > 0 else int(sec_data) + 1

    runs_count = num_flows * len(freqs)
    therms = ["Proximal", "Distal"] + (["Ivan"] if use_ivan else [])
    cols_per_run = len(therms)
    total_columns = runs_count * cols_per_run
    approx_rows = expected_samples

    filename = format_filename(prefix="flowdata")
    print(f"\nData will be saved to: {filename}")
    print(f"Planned layout: {approx_rows} sample rows x {total_columns} columns "
          f"({runs_count} runs × {cols_per_run} thermometers per run)\n")
    print(f"Top cell of each column will contain metadata like: 'Dist,10ms,1.0Hz,6.0GPM'\n")

    # configure ESP (ivan and interval)
    ser.write(f"ivan:{1 if use_ivan else 0}\n".encode())
    ser.write(f"interval:{interval_ms}\n".encode())
    time.sleep(0.05)

    # prepare storage
    columns = []
    metadata_cells = []

    with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)

        try:
            for flow_idx in range(num_flows):
                flow = float(read_input(f"\n--- Test Run #{flow_idx+1} ---\nEnter current flow reading: ", "0.0", float))
                for freq in freqs:
                    # set freq and announce
                    ser.write(f"freq:{freq}\n".encode())
                    time.sleep(0.05)

                    # flush any backlog BEFORE starting the stream so we don't later discard fresh stream samples
                    flush_input_buffer_safe(ser)

                    # start streaming (ESP will now begin sending temp samples)
                    ser.write("start_stream\n".encode())

                    # Invalidate and ignore only a short burst to avoid stale/saturated sample(s)
                    reader.invalidate()
                    # small ignore period to drop any immediate glitch samples (keep short so you don't miss data)
                    ignore_seconds = max(0.05, (interval_ms / 1000.0) * 1.0)
                    reader.ignore_for(ignore_seconds)

                    # allow first fresh sample(s) before prompting
                    time.sleep(max(0.02, interval_ms / 1000.0))

                    # IMPORTANT: display oscillation message and wait for the user to press Enter BEFORE recording
                    prompt = f"Oscillating at {freq} Hz. Press Enter when temperatures stabilize to BEGIN RECORDING."
                    wait_for_enter_with_timer(prompt)

                    # User pressed Enter -> BEGIN RECORDING
                    print(f"\nRecording now, please wait {sec_data} seconds...")
                    # begin recording (reader appends raw samples as they arrive)
                    reader.start_recording()

                    # record for exactly sec_data seconds (wall clock)
                    start_t = time.time()
                    end_t = start_t + sec_data
                    # loop with a short sleep to be responsive if you want to interrupt later
                    try:
                        while time.time() < end_t:
                            time.sleep(0.01)
                    except KeyboardInterrupt:
                        print("\nRecording interrupted by user.")
                    finally:
                        reader.stop_recording()

                    # small pause to ensure last samples processed
                    time.sleep(0.02)
                    recorded = reader.get_recorded_copy()

                    # truncate/pad as needed
                    if len(recorded) < expected_samples:
                        print(f"\nWarning: Collected {len(recorded)} samples (expected {expected_samples}); padding with empty values.")
                    if len(recorded) > expected_samples:
                        recorded = recorded[:expected_samples]

                    # build per-therm columns
                    prox_col = []
                    dist_col = []
                    ivan_col = [] if use_ivan else None
                    for sample in recorded:
                        prox_col.append(sample[0] if len(sample) >= 1 else "")
                        dist_col.append(sample[1] if len(sample) >= 2 else "")
                        if use_ivan:
                            ivan_col.append(sample[2] if len(sample) >= 3 else "")

                    # pad to expected_samples
                    while len(prox_col) < expected_samples:
                        prox_col.append("")
                    while len(dist_col) < expected_samples:
                        dist_col.append("")
                    if use_ivan:
                        while len(ivan_col) < expected_samples:
                            ivan_col.append("")

                    # append columns in order: Proximal, Distal, (Ivan)
                    columns.append(prox_col)
                    columns.append(dist_col)
                    if use_ivan:
                        columns.append(ivan_col)

                    # metadata
                    dt_ms = interval_ms
                    metadata_cells.append(f"Proximal,{dt_ms}ms,{freq}Hz,{flow}GPM")
                    metadata_cells.append(f"Distal,{dt_ms}ms,{freq}Hz,{flow}GPM")
                    if use_ivan:
                        metadata_cells.append(f"Ivan,{dt_ms}ms,{freq}Hz,{flow}GPM")

                    # a few blank lines then summary for readability
                    print("\n\n")
                    print(f"Finished recording for freq {freq} Hz at flow {flow} GPM.")
                    print(f"Samples recorded: {len(recorded)} (expected {expected_samples})")
                    print("\n\n")

                    ser.write("stop_stream\n".encode())
                    time.sleep(0.05)
                    time.sleep(0.15)

            # adjust columns count if needed
            if len(columns) != total_columns:
                print(f"\nNote: planned {total_columns} columns but actually have {len(columns)} columns; writing what we have.")
                total_columns = len(columns)

            # write metadata row
            writer.writerow(metadata_cells)

            # write sample rows
            for r in range(expected_samples):
                row = []
                for c in range(total_columns):
                    row.append(columns[c][r] if r < len(columns[c]) else "")
                writer.writerow(row)
            f.flush()

            print(f"\nAll runs complete. File saved: {filename}")
            print(f"File contains: {expected_samples} sample rows + 1 metadata row = {expected_samples+1} total rows, {total_columns} columns.")

        finally:
            try:
                ser.write("stop_stream\n".encode())
            except:
                pass
            reader.stop()
            reader.join(timeout=1.0)
            ser.close()


if __name__ == "__main__":
    main()
