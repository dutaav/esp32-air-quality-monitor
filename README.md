# Air Quality Monitoring

An **air quality** monitoring system that runs entirely on the ESP32. The sensors read gas, temperature, and humidity, then the system computes an **AQI** and **predicts the next AQI value** using **linear regression** over **time-series features** — all processed on-device, no internet required.

The approach: a hand-rolled time-series pipeline (MQ-135 → moving average → AQI → delta/acceleration features) feeding a linear-regression model whose weights are trained offline and embedded into the firmware.

---

## Table of Contents

- [System Flow](#system-flow)
- [Highlights](#highlights)
- [Hardware](#hardware)
- [Wiring](#wiring)
- [Arduino Libraries](#arduino-libraries)
- [Folder Structure](#folder-structure)
- [Dataset](#dataset)
- [How to Run](#how-to-run)
- [Model & Evaluation](#model--evaluation)
- [Algorithm Details](#algorithm-details)
- [Output & Status](#output--status)
- [Blynk Dashboard](#blynk-dashboard-optional)

---

## System Flow

```
[ SENSORS ]     MQ-135 (gas)  +  DHT22 (temp/RH)
     |
     v
[ ACQUISITION ] read 16x burst -> denoise
     |
     v
[ ALGORITHM ]   circular buffer (moving average) -> temp/RH correction -> AQI
     |          + time-series features: delta & acceleration
     v
[ AI MODEL ]    linear regression (7 features) -> predict next AQI
     |
     v
[ OUTPUT ]      OLED + 2 LEDs + buzzer + Serial CSV + Blynk (optional)
```

One cycle runs every **2 seconds** (`PERIOD_MS = 2000`), non-blocking.

---

## Highlights

| Area | What it does | Location |
|---|---|---|
| **Algorithm (from scratch)** | `O(1)` circular buffer for moving average + time-series feature extraction (delta = AQI difference, acceleration = second difference) | [esp.ino](esp.ino) |
| **AI/ML** | Multivariate linear regression with 7 features. Trained offline on a laptop (least-squares / SVD), `W_*` coefficients embedded into the firmware | [model/train_model.py](model/train_model.py) |
| **Hardware** | ESP32 + MQ-135 + DHT22 + OLED + 2 LEDs + buzzer | [Wiring](#wiring) |

Extras: residual-based **anomaly detection** (large prediction errors in a row) and a **fire alarm** (temperature ≥ 50 °C). Optional **Blynk** dashboard for remote monitoring — the device still runs **fully offline** if WiFi/Blynk is unavailable.

---

## Hardware

| Component | Function |
|---|---|
| **ESP32 DevKit** | Main microcontroller (12-bit ADC, 0–4095) |
| **MQ-135** | Gas / air quality sensor (CO₂, smoke, VOC, etc.) |
| **DHT22** | Temperature & humidity sensor (for gas reading correction) |
| **OLED SSD1306 128×64** | I²C display (address `0x3C`) |
| **Green LED + Red LED** | Status indicators (2 LEDs; yellow simulated by green+red) |
| **Active buzzer** | Audible alarm (bad air / fire / anomaly) |

---

## Wiring

| Device | Device Pin | ESP32 Pin |
|---|---|---|
| MQ-135 | AOUT (analog) | **GPIO35** |
| MQ-135 | DOUT (digital) | **GPIO34** |
| DHT22 | DATA | **GPIO4** |
| Green LED | + | **GPIO12** |
| Red LED | + | **GPIO13** |
| Buzzer | + | **GPIO14** |
| OLED | SDA | **GPIO21** (default I²C) |
| OLED | SCL | **GPIO22** (default I²C) |

> Note: GPIO34 & GPIO35 are **input-only** pins on the ESP32, ideal for analog sensor reads.

---

## Arduino Libraries

Install via **Library Manager** in the Arduino IDE:

- `Adafruit GFX Library`
- `Adafruit SSD1306`
- `DHT sensor library` (Adafruit)
- `Wire` (built-in, I²C)
- `Blynk` (by Volodymyr Shymanskyy — for the optional dashboard)
- `WiFi` (built-in with the ESP32 core)

Board: **ESP32 Dev Module** (*esp32 by Espressif Systems* package).

---

## Folder Structure

```
├── esp.ino              # ESP32 firmware (acquisition, algorithm, regression, output)
├── dataset/
│   └── cleaned.csv      # Cleaned data + time-series features (ready to train)
└── model/
    ├── record_serial.py # Record Serial CSV, clean in memory -> cleaned.csv
    ├── train_model.py   # Train linear regression (SVD) -> print W_* coefficients
    └── plot_model.py    # Visualization: actual vs predicted, residuals, feature weights
```

---

## Dataset

**Data source: recorded directly from our own ESP32.** The firmware [esp.ino](esp.ino) prints a CSV line to Serial every cycle, and [record_serial.py](model/record_serial.py) captures it, cleans it in memory, and writes the ready-to-train `cleaned.csv`

Column format (header from the firmware):

```
Time,Temp,Humidity,RawGas,CorrectedGas,AQI,PredictedAQI
```

| Column | Meaning |
|---|---|
| `Time` | Seconds since the ESP32 powered on |
| `Temp` / `Humidity` | Temperature (°C) & humidity (%) from DHT22 |
| `RawGas` | MQ-135 ADC reading (moving-average smoothed) |
| `CorrectedGas` | Gas after temp/RH correction factor |
| `AQI` | Computed AQI (0–500 scale) |
| `PredictedAQI` | Firmware's AQI prediction (for evaluation) |

**Automatic cleaning** ([record_serial.py](model/record_serial.py) → `clean()`):
- Drop repeated headers & force numeric
- Filter to physically valid ranges (temp 0–60 °C, RH 0–100%, AQI 5–500)
- Remove adaptive outliers (**IQR** method) on `AQI` & `RawGas`
- Sort by `Time`, then build time-series features:
  `AQI_prev1`, `AQI_prev2`, `delta_AQI`, `delta2_AQI`

Current dataset: `cleaned.csv`.

---

## How to Run

### 1. Flash the firmware

Open [esp.ino](esp.ino) in the Arduino IDE → select **ESP32 Dev Module** → Upload (Serial baud **115200**).

### 2. Record a dataset (optional, if you want your own data)

```bash
# Python venv with numpy, pandas, matplotlib, pyserial
python model/record_serial.py /dev/cu.usbserial-0001 600
```

Records for 600 seconds (10 minutes) then **auto-stops**. Close the Arduino Serial Monitor first so the port isn't busy. Press `Ctrl-C` to stop early (data still gets cleaned). Output: `dataset/cleaned.csv` (captured and cleaned in one step, no intermediate file).

> Tip: vary the air while recording — breathe on the sensor, bring smoke/perfume close, then let it recover — so the AQI swings up and down and the model learns the pattern.

### 3. Train the model

```bash
python model/train_model.py
```

Prints `const float W_*` coefficients ready to **paste into [esp.ino](esp.ino)** (the *Regression weights* block).

### 4. View analysis (optional)

```bash
python model/plot_model.py
```

Generates `model/model_analysis.png` (actual vs predicted, residuals, feature weights).

---

## Model & Evaluation

The linear regression is trained with **least-squares via SVD** (`np.linalg.lstsq`).

> Why SVD instead of the normal equation? `delta_AQI` & `delta2_AQI` are linear combinations of AQI lags → the `XᵀX` matrix is *rank-deficient*. SVD yields a stable *minimum-norm* solution.

**Performance (test set, sequential 80/20 split):**

| Metric | Value |
|---|---|
| Train / test rows | 718 / 180 |
| **MAE** | **1.46** AQI units |
| **RMSE** | **1.88** |
| **R²** | **0.9974** |

**Trained coefficients (embedded in the firmware):**

| Feature | Weight |
|---|---|
| `W_AQI` | +0.93774 |
| `W_PREV1` | +0.32736 |
| `W_PREV2` | −0.27021 |
| `W_DELTA` | +0.61038 |
| `W_ACCEL` | +0.01281 |
| `W_TEMP` | +0.01409 |
| `W_HUM` | −0.22247 |
| `W_BIAS` | +17.14996 |

---

## Algorithm Details

**Circular buffer.** Moving average of gas readings with a window of 10, updated in `O(1)`: subtract the old value, add the new one, advance the head — no re-looping.

```
prediction = W_AQI·AQI + W_PREV1·prev1 + W_PREV2·prev2
           + W_DELTA·delta + W_ACCEL·accel
           + W_TEMP·temp + W_HUM·RH + W_BIAS
```

- **delta** = `AQI − AQI_prev1` (rate of change)
- **accel** = `delta − prev_delta` (second difference, trend acceleration)

The feature math in the firmware is **identical** to what `record_serial.py` builds, so the trained coefficients stay consistent with runtime.

**On-device accuracy.** A 30-sample error ring buffer → MAE → `accuracy = 100 − MAE/5`, shown directly on the OLED.

**Complexity.** Acquisition & prediction are `O(1)` per cycle (fixed window), constant memory — safe for the ESP32's RAM.

---

## Output & Status

| Status | Condition | LED | Buzzer |
|---|---|---|---|
| **GOOD** | AQI ≤ 100 | Green | Off |
| **MODERATE** | 100 < AQI ≤ 300 | Green + Red | Short beep |
| **BAD** | AQI > 300 | Red | Continuous |
| **FIRE!** | Temp ≥ 50 °C | Red | Fast beeps |
| **ANOMALY** | Large prediction error for 3 cycles | (running status) | Double beep |

**Edge-case handling:**
- **20-second warm-up** — wait for the MQ-135 heater to stabilize before starting
- **DHT error / NaN** — reuse the last valid reading
- **Buffer not full** — show fill progress, hold output
- **Gas rail** (ADC ≤ 5 or ≥ 4090) — flag `gasERR`
- **First-cycle delta spike** — seed `prev` values with the initial AQI

---

## Blynk Dashboard (optional)

The firmware can stream its telemetry to a **Blynk IoT** dashboard for remote monitoring. This is **purely additive**: all the AI/algorithm work still runs on-device, and the system keeps working even if WiFi or the Blynk server is down (non-blocking connect with periodic retry).

> A small filled **dot appears at the top-right of the OLED** when Blynk is connected.

**Virtual datastreams** (set these up in the Blynk console):

| Pin | Datastream | Type | Range |
|---|---|---|---|
| `V0` | AQI | Integer | 0–500 |
| `V1` | Predicted AQI | Integer | 0–500 |
| `V2` | Temperature | Double | 0–60 °C |
| `V3` | Humidity | Double | 0–100 % |
| `V4` | Corrected gas | Double | — |
| `V5` | Status | String | GOOD / MODERATE / BAD / FIRE! |
| `V6` | Accuracy | Integer | 0–100 % |
| `V7` | Trend | String | Up / Down / Stable |

**Events** (Blynk console → *Events*, used for push notifications):

| Event code | Trigger |
|---|---|
| `fire_alarm` | Temperature ≥ 50 °C (rising edge) |
| `anomaly` | Large prediction error for 3 cycles (rising edge) |

**Setup:**

1. Install the **`Blynk`** library (Library Manager).
2. In the [Blynk console](https://blynk.cloud), create a template, add the datastreams/events above, then grab the device's `BLYNK_TEMPLATE_ID`, `BLYNK_TEMPLATE_NAME`, and `BLYNK_AUTH_TOKEN`.
3. Paste those three values + your `WIFI_SSID` / `WIFI_PASS` into the Blynk block near the top of [esp.ino](esp.ino).
4. Upload. The device connects during the warm-up window; leave the fields as-is to run offline-only.

> Serial debug from Blynk is intentionally **disabled** (`BLYNK_PRINT` undefined) so the CSV stream stays clean for [record_serial.py](model/record_serial.py).

---

*All core computation (algorithm + AI) runs entirely on the ESP32. The cloud is optional — Blynk only mirrors the results for remote viewing, and the device stays fully functional offline.*
