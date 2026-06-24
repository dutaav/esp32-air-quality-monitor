// Air Quality Monitoring - ESP32
// Time-series (circular buffer + delta/acceleration) + linear regression.
// Weights trained offline (model/latih_model.py), embedded here.

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <DHT.h>

// ---- Blynk IoT (template IDs MUST be defined before the Blynk header) ----
// BLYNK_PRINT intentionally left undefined: keeps Serial clean for the CSV logger.
#define BLYNK_TEMPLATE_ID   "TMPL6C8L3fR8n"
#define BLYNK_TEMPLATE_NAME "Air Quality"
#define BLYNK_AUTH_TOKEN    "LY2hbYvSKWOuYESDebySWZsRt6OTir1-"
#include <WiFi.h>
#include <BlynkSimpleEsp32.h>

// WiFi credentials. Device still runs fully offline if these are wrong/empty.
const char* WIFI_SSID = "Glu";
const char* WIFI_PASS = "G11111111";

// Blynk virtual datastream pins
#define VP_AQI    V0
#define VP_PRED   V1
#define VP_TEMP   V2
#define VP_HUM    V3
#define VP_GAS    V4
#define VP_STATUS V5
#define VP_ACC    V6
#define VP_TREND  V7

#define PIN_GAS_AOUT 35
#define PIN_GAS_DOUT 34
#define PIN_DHT      4
#define PIN_GREEN    12
#define PIN_RED      13
#define PIN_BUZZER   14

#define OLED_W    128
#define OLED_H    64
#define OLED_ADDR 0x3C
Adafruit_SSD1306 oled(OLED_W, OLED_H, &Wire, -1);

DHT dht(PIN_DHT, DHT22);

#define ADC_RES 12
#define ADC_MAX 4095

const unsigned long PERIOD_MS = 2000;   // matches dataset Time cadence
const unsigned long WARMUP_MS = 20000;  // MQ-135 heater stabilization

// MQ-135 correction factors (datasheet approximation)
const float TEMP_REF = 20.0f, HUM_REF = 65.0f;
const float COEF_TEMP = 0.02f, COEF_HUM = 0.008f;

// correctedGas -> AQI linear map; must match training dataset scale
const float GAS_CLEAN = 200.0f;
const float GAS_BAD   = 800.0f;
const int   AQI_MAX   = 500;

const int   THRESH_GOOD = 100;
const int   THRESH_MOD  = 300;
const float TEMP_FIRE   = 50.0f;

// Regression weights (features: AQI, prev1, prev2, delta, accel, temp, humidity)
const float W_AQI = 0.93774f, W_PREV1 = 0.32736f, W_PREV2 = -0.27021f;
const float W_DELTA = 0.61038f, W_ACCEL = 0.01281f;
const float W_TEMP = 0.01409f, W_HUM = -0.22247f, W_BIAS = 17.14996f;

// --- Circular buffer for O(1) moving average of gas readings ---
#define WINDOW 10
int  bufSlots[WINDOW];
int  bufHead = 0, bufCount = 0;
long bufSum  = 0;

void maPush(int val) {
  bufSum -= bufSlots[bufHead];
  bufSlots[bufHead] = val;
  bufSum += val;
  bufHead = (bufHead + 1) % WINDOW;
  if (bufCount < WINDOW) bufCount++;
}
int maMean() { return bufCount ? bufSum / bufCount : 0; }

// --- Error ring buffer: tracks MAE for accuracy estimate ---
#define ERR_RING 30
int errRing[ERR_RING];
int errIdx = 0, errCount = 0;

void errAdd(int e) {
  errRing[errIdx] = e;
  errIdx = (errIdx + 1) % ERR_RING;
  if (errCount < ERR_RING) errCount++;
}
int calcMae() {
  if (!errCount) return 0;
  long s = 0;
  for (int i = 0; i < errCount; i++) s += errRing[i];
  return s / errCount;
}

// --- Sensor state & time-series variables ---
int  aqi = 0, aqiPrev1 = 0, aqiPrev2 = 0;
int  delta = 0, accel = 0;
int  predicted = 0, prevPredicted = 0, accuracy = 0;
bool seriesReady = false;

float temp = 0.0f, humidity = 0.0f, correctedGas = 0.0f;
int   gasAvg = 0;
bool  dhtErr = false, gasErr = false;
const char* trend = "Stable";

bool fireAlarm = false;
unsigned long tBoot = 0, tCycle = 0;
long secElapsed = 0;

// --- Anomaly detection (residual-based) ---
const int ANOMALY_THRESH = 25;   // |error| threshold
const int ANOMALY_CYCLES = 3;    // consecutive cycles to trigger
int  anomalyCount = 0;
bool anomaly      = false;

// Non-blocking buzzer
enum BuzzerMode { BUZ_OFF, BUZ_SHORT, BUZ_CONT, BUZ_FIRE, BUZ_ANOMALY };
BuzzerMode    buzzerMode = BUZ_OFF;
unsigned long tBuzzer = 0;
bool          buzzerToggle = false;

// Blynk: non-blocking reconnect timer + alert edge tracking
const unsigned long BLYNK_RETRY_MS = 5000;
unsigned long tBlynkRetry = 0;
bool prevFire = false, prevAnomaly = false;

void setLed(bool green, bool red) {
  digitalWrite(PIN_GREEN, green);
  digitalWrite(PIN_RED, red);
}

void serviceBuzzer() {
  switch (buzzerMode) {
    case BUZ_OFF:   digitalWrite(PIN_BUZZER, LOW);  break;
    case BUZ_CONT:  digitalWrite(PIN_BUZZER, HIGH); break;
    case BUZ_SHORT: digitalWrite(PIN_BUZZER, millis() - tBuzzer < 150); break;
    case BUZ_FIRE:
      if (millis() - tBuzzer >= 120) {
        tBuzzer = millis();
        buzzerToggle = !buzzerToggle;
        digitalWrite(PIN_BUZZER, buzzerToggle);
      }
      break;
    case BUZ_ANOMALY: {   // double beep: 80ms on, 80ms off, 80ms on
        unsigned long dt = millis() - tBuzzer;
        digitalWrite(PIN_BUZZER, (dt < 80) || (dt >= 160 && dt < 240));
      }
      break;
  }
}

// --- Sensor acquisition ---
int readRawGas() {          // burst-16 average to reduce noise
  long s = 0;
  for (int i = 0; i < 16; i++) { s += analogRead(PIN_GAS_AOUT); delay(2); }
  return s / 16;
}

void readDht() {            // on NaN, keep last valid reading
  float t = dht.readTemperature(), h = dht.readHumidity();
  dhtErr = isnan(t) || isnan(h);
  if (!dhtErr) { temp = t; humidity = h; }
}

float correctGas(int raw) {
  float f = 1.0f + COEF_TEMP * (temp - TEMP_REF) + COEF_HUM * (humidity - HUM_REF);
  if (f < 0.1f) f = 0.1f;
  return raw / f;
}

int gasToAqi(float corrected) {
  float a = (corrected - GAS_CLEAN) * AQI_MAX / (GAS_BAD - GAS_CLEAN);
  return constrain(lroundf(a), 0, AQI_MAX);
}

// --- Time-series features + linear regression ---
int predictAqi() {
  int prevDelta = aqiPrev1 - aqiPrev2;
  delta = aqi - aqiPrev1;
  accel = delta - prevDelta;
  float p = W_AQI * aqi + W_PREV1 * aqiPrev1 + W_PREV2 * aqiPrev2
          + W_DELTA * delta + W_ACCEL * accel
          + W_TEMP * temp + W_HUM * humidity + W_BIAS;
  return constrain(lroundf(p), 0, AQI_MAX);
}

const char* airStatus() {
  if (fireAlarm)                 return "FIRE!";
  if (aqi <= THRESH_GOOD)        return "GOOD";
  if (aqi <= THRESH_MOD)         return "MOD";
  return "BAD";
}

// 2 LEDs: GOOD=green, MOD=green+red (fake yellow)+short beep, BAD=red+continuous
void controlOutputs() {
  if (fireAlarm)               { setLed(false, true); buzzerMode = BUZ_FIRE; return; }
  if (aqi <= THRESH_GOOD)      { setLed(true,  false); buzzerMode = BUZ_OFF; }
  else if (aqi <= THRESH_MOD)  { setLed(true,  true);  buzzerMode = BUZ_SHORT; tBuzzer = millis(); }
  else                         { setLed(false, true);  buzzerMode = BUZ_CONT; }
  if (anomaly && buzzerMode < BUZ_CONT) { buzzerMode = BUZ_ANOMALY; tBuzzer = millis(); }
}

// --- OLED display: single function, phase-driven ---
enum Phase { PH_WARMUP, PH_FILL, PH_FIRE, PH_NORMAL };

void showDisplay(Phase phase, int secsLeft = 0) {
  oled.clearDisplay();
  oled.setTextColor(SSD1306_WHITE);
  oled.setTextSize(1);
  oled.setCursor(0, 0);

  if (phase == PH_WARMUP) {
    oled.println("MQ-135 Warming Up");
    oled.drawLine(0, 10, 128, 10, SSD1306_WHITE);
    oled.setCursor(0, 24); oled.print("Wait "); oled.print(secsLeft); oled.println(" s");
    oled.setCursor(0, 44); oled.println("Stabilizing...");
    oled.display(); return;
  }

  if (phase == PH_FILL) {
    oled.println("Air Quality");
    oled.drawLine(0, 10, 128, 10, SSD1306_WHITE);
    oled.setCursor(0, 26);
    oled.print("Filling "); oled.print(bufCount); oled.print("/"); oled.println(WINDOW);
    oled.display(); return;
  }

  if (phase == PH_FIRE) {
    oled.setTextSize(2); oled.setCursor(8, 6); oled.println("FIRE!!");
    oled.setTextSize(1);
    oled.setCursor(0, 34); oled.print("Temp "); oled.print(temp, 1); oled.println(" C");
    oled.setCursor(0, 48); oled.println("Evacuate now!");
    oled.display(); return;
  }

  // PH_NORMAL
  oled.print("Air Quality");
  if (Blynk.connected()) oled.fillCircle(124, 3, 3, SSD1306_WHITE);  // online dot
  oled.drawLine(0, 10, 128, 10, SSD1306_WHITE);
  oled.setTextSize(2); oled.setCursor(0, 14); oled.print(aqi);
  oled.setTextSize(1); oled.setCursor(72, 14); oled.print(airStatus());
  if (gasErr) { oled.setCursor(72, 24); oled.print("gasERR"); }
  oled.setCursor(0, 34); oled.print("T:");
  dhtErr ? oled.print("--") : oled.print(temp, 1);
  oled.print(" H:");
  dhtErr ? oled.print("--") : oled.print(humidity, 0);
  oled.setCursor(0, 44); oled.print("Pred:"); oled.print(predicted); oled.print(" "); oled.print(trend);
  if (anomaly) {
    oled.setCursor(0, 54); oled.print("!! ANOMALY !!");
  } else {
    oled.setCursor(0, 54); oled.print("Acc:"); oled.print(accuracy); oled.print("%");
  }
  oled.display();
}

void printCsv() {
  Serial.print(secElapsed);       Serial.print(',');
  Serial.print(temp, 1);          Serial.print(',');
  Serial.print(humidity, 1);      Serial.print(',');
  Serial.print(gasAvg);           Serial.print(',');
  Serial.print(correctedGas, 1);  Serial.print(',');
  Serial.print(aqi);              Serial.print(',');
  Serial.println(predicted);
}

// --- Blynk: non-blocking connect + telemetry push (edge stays usable offline) ---
void blynkSetup() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Blynk.config(BLYNK_AUTH_TOKEN);   // non-blocking; blynkService() drives the rest
}

void blynkService() {               // called every loop(); never blocks the cycle
  if (WiFi.status() != WL_CONNECTED) {
    if (millis() - tBlynkRetry >= BLYNK_RETRY_MS) {
      tBlynkRetry = millis();
      WiFi.begin(WIFI_SSID, WIFI_PASS);
    }
    return;
  }
  if (!Blynk.connected()) {
    if (millis() - tBlynkRetry >= BLYNK_RETRY_MS) {
      tBlynkRetry = millis();
      Blynk.connect(1500);          // short timeout so a dead server can't hang us
    }
    return;
  }
  Blynk.run();
}

void pushBlynk() {                  // datastream values + edge-triggered alerts
  bool fireEdge    = fireAlarm && !prevFire;
  bool anomalyEdge = anomaly   && !prevAnomaly;
  prevFire    = fireAlarm;
  prevAnomaly = anomaly;

  if (!Blynk.connected()) return;
  Blynk.virtualWrite(VP_AQI,    aqi);
  Blynk.virtualWrite(VP_PRED,   predicted);
  Blynk.virtualWrite(VP_TEMP,   temp);
  Blynk.virtualWrite(VP_HUM,    humidity);
  Blynk.virtualWrite(VP_GAS,    correctedGas);
  Blynk.virtualWrite(VP_STATUS, airStatus());
  Blynk.virtualWrite(VP_ACC,    accuracy);
  Blynk.virtualWrite(VP_TREND,  trend);
  if (fireEdge)    Blynk.logEvent("fire_alarm");
  if (anomalyEdge) Blynk.logEvent("anomaly");
}

void runCycle() {
  secElapsed = millis() / 1000;

  maPush(readRawGas());
  gasAvg = maMean();
  gasErr = (gasAvg <= 5) || (gasAvg >= ADC_MAX - 5);
  readDht();

  // Warmup: keep outputs off until MQ-135 stabilizes
  if (millis() - tBoot < WARMUP_MS) {
    int secs = (WARMUP_MS - (millis() - tBoot) + 999) / 1000;
    setLed(false, false); buzzerMode = BUZ_OFF;
    showDisplay(PH_WARMUP, secs);
    return;
  }
  // Fill: wait until moving average window is full
  if (bufCount < WINDOW) {
    setLed(false, false); buzzerMode = BUZ_OFF;
    showDisplay(PH_FILL);
    return;
  }

  correctedGas = correctGas(gasAvg);
  aqi = gasToAqi(correctedGas);

  if (!seriesReady) {   // prevent delta spike on first cycle
    aqiPrev1 = aqiPrev2 = prevPredicted = aqi;
    seriesReady = true;
  }

  errAdd(abs(prevPredicted - aqi));
  accuracy = constrain(100 - calcMae() / 5, 0, 100);

  predicted     = predictAqi();
  prevPredicted = predicted;

  int diff = predicted - aqi;
  trend = (diff > 5) ? "Up" : (diff < -5) ? "Down" : "Stable";

  // Anomaly: large prediction error for N consecutive cycles
  int cycleErr = abs(prevPredicted - aqi);
  if (cycleErr > ANOMALY_THRESH) {
    anomalyCount++;
    anomaly = (anomalyCount >= ANOMALY_CYCLES);
  } else {
    anomalyCount = 0;
    anomaly = false;
  }

  fireAlarm = !dhtErr && temp >= TEMP_FIRE;

  controlOutputs();
  showDisplay(fireAlarm ? PH_FIRE : PH_NORMAL);
  printCsv();
  pushBlynk();

  aqiPrev2 = aqiPrev1;
  aqiPrev1 = aqi;
}

void setup() {
  Serial.begin(115200);
  Serial.println("Time,Temp,Humidity,RawGas,CorrectedGas,AQI,PredictedAQI");

  pinMode(PIN_GAS_DOUT, INPUT);
  pinMode(PIN_GREEN,    OUTPUT);
  pinMode(PIN_RED,      OUTPUT);
  pinMode(PIN_BUZZER,   OUTPUT);
  analogReadResolution(ADC_RES);

  dht.begin();
  if (!oled.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println("OLED not found");
    while (true) delay(1000);
  }

  setLed(true, true); digitalWrite(PIN_BUZZER, HIGH); delay(500);
  setLed(false, false); digitalWrite(PIN_BUZZER, LOW);

  oled.clearDisplay(); oled.setTextColor(SSD1306_WHITE);
  oled.setTextSize(1); oled.setCursor(0, 0);
  oled.println("Air Quality");
  oled.println("Monitoring");
  oled.println("ESP32 + Linear");
  oled.println("Regression");
  oled.display();
  delay(1200);

  blynkSetup();   // starts connecting during the MQ-135 warm-up window

  tBoot  = millis();
  tCycle = millis();
}

void loop() {
  serviceBuzzer();
  blynkService();
  if (millis() - tCycle >= PERIOD_MS) {
    tCycle += PERIOD_MS;
    runCycle();
  }
}
