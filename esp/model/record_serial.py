#!/usr/bin/env python3
import os
import sys
import time
import glob
import serial
import pandas as pd

BAUD = 115200
HEADER = "Time,Temp,Humidity,RawGas,CorrectedGas,AQI,PredictedAQI"
COLUMNS = HEADER.split(",")
DEFAULT_SECONDS = 600.0   # 10 minutes, then auto-stop

DIR_HERE     = os.path.dirname(os.path.abspath(__file__))
DIR_DATA     = os.path.join(DIR_HERE, "..", "dataset")
PATH_RAW     = os.path.join(DIR_DATA, "raw.csv")
PATH_CLEANED = os.path.join(DIR_DATA, "cleaned.csv")


def find_port(arg: str | None) -> str:
    if arg:
        return arg
    for pat in ("/dev/cu.usbserial*", "/dev/cu.wchusbserial*",
                "/dev/cu.SLAB*", "/dev/cu.usbmodem*"):
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    sys.exit("Port not found. Pass it manually: python record_serial.py /dev/cu.xxxx")


def record(port: str, seconds: float, path_raw: str) -> int:
    """Stream CSV lines from serial into path_raw. Returns the row count."""
    ser = serial.Serial(port, BAUD, timeout=1)
    ser.reset_input_buffer()                       # drop stale OS buffer backlog

    print(f"Recording {port} @ {BAUD} -> {path_raw}  (auto-stop {seconds:.0f}s)")
    print("Vary the air: breathe on the sensor / bring smoke or perfume, then let it recover.\n")

    t0 = time.time()
    n = 0
    with open(path_raw, "w") as f:
        f.write(HEADER + "\n")
        try:
            while time.time() - t0 < seconds:
                line = ser.readline().decode("utf-8", "ignore").strip()
                if not line:
                    continue
                if line.startswith("#"):           # baseline comment -> show only
                    print("  " + line)
                    continue
                if line.startswith("Time"):        # stream header -> already written
                    continue
                if not line[0].isdigit():          # skip partial/other text
                    continue
                f.write(line + "\n")
                f.flush()
                n += 1
                left = seconds - (time.time() - t0)
                print(f"[left {left:5.0f}s] ({n:4d}) {line}")
        except KeyboardInterrupt:
            print("\nStopped early (Ctrl-C).")
        finally:
            ser.close()
    return n


def drop_outliers_iqr(df: pd.DataFrame, column: str, k: float = 1.5) -> pd.DataFrame:
    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return df[(df[column] >= lo) & (df[column] <= hi)]


def clean(path_raw: str, path_cleaned: str) -> pd.DataFrame:
    """Clean raw CSV and add time-series features. Returns the cleaned frame."""
    df = pd.read_csv(path_raw)
    if not set(COLUMNS).issubset(df.columns):
        raise ValueError(f"Missing columns. Need: {COLUMNS}")

    # drop repeated headers, force numeric, keep physically valid ranges
    df = df[df["Time"].astype(str).str.lower() != "time"]
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df = df[df["Temp"].between(0, 60)]
    df = df[df["Humidity"].between(0, 100)]
    df = df[df["CorrectedGas"] > 0]
    df = df[(df["AQI"] > 5) & (df["AQI"] < 500)]
    df = df[df["PredictedAQI"] < 500]

    # adaptive outlier removal (robust across rooms/sensors)
    df = drop_outliers_iqr(df, "AQI")
    df = drop_outliers_iqr(df, "RawGas")

    # order by time before computing lag/delta features
    df = df.sort_values("Time").reset_index(drop=True)

    # time-series features (must match the math in esp.ino)
    df["AQI_prev1"]  = df["AQI"].shift(1)
    df["AQI_prev2"]  = df["AQI"].shift(2)
    df["delta_AQI"]  = df["AQI"] - df["AQI_prev1"]
    df["delta2_AQI"] = df["delta_AQI"] - df["delta_AQI"].shift(1)

    df = df.dropna().reset_index(drop=True)
    df.to_csv(path_cleaned, index=False)
    return df


def main() -> None:
    port: str | None = None
    seconds = DEFAULT_SECONDS
    for a in sys.argv[1:]:
        if a.replace(".", "", 1).isdigit():
            seconds = float(a)
        else:
            port = a
    port = find_port(port)

    n = record(port, seconds, PATH_RAW)
    if n < 5:
        sys.exit(f"\nOnly {n} rows recorded, too few to train. "
                 "Make sure the ESP32 streams CSV, then try again for longer.")

    print(f"\nRecording done: {n} raw rows -> {PATH_RAW}")
    print("Cleaning + building time-series features...")
    df = clean(PATH_RAW, PATH_CLEANED)
    print(f"Cleaned: {len(df)} rows -> {PATH_CLEANED}")
    print("\nNow train: python train_model.py")


if __name__ == "__main__":
    try:
        main()
    except serial.SerialException as e:
        sys.exit(f"Serial error: {e}\n-> Port busy? Close the Arduino IDE Serial Monitor first.")
