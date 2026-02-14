// ESP32 v3 bridge firmware skeleton
// Features: CRC check, watchdog(200ms), ramp down(300ms), ackermann steer mapping
#include <Arduino.h>

static const uint8_t SOF0 = 0xAA;
static const uint8_t SOF1 = 0x55;
static const uint8_t FRAME_CMD = 0x02;

static const uint32_t WATCHDOG_MS = 200;
static const uint32_t RAMP_MS = 300;

float cmd_vl = 0, cmd_vr = 0, cmd_steer = 0;
float out_vl = 0, out_vr = 0;
uint32_t last_cmd_ms = 0;

uint16_t crc16_ccitt(const uint8_t *data, size_t len) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < len; ++i) {
    crc ^= (uint16_t)data[i] << 8;
    for (int b = 0; b < 8; ++b) {
      crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
    }
  }
  return crc;
}

bool read_frame() {
  static uint8_t buf[64];
  if (Serial.available() < 6) return false;
  if (Serial.read() != SOF0) return false;
  if (Serial.read() != SOF1) return false;
  uint8_t type = Serial.read();
  uint8_t len = Serial.read();
  if (type != FRAME_CMD || len > 48) return false;
  while (Serial.available() < len + 2) return false;
  Serial.readBytes((char*)buf, len + 2);
  uint16_t recv = (uint16_t)buf[len] | ((uint16_t)buf[len + 1] << 8);
  uint8_t hdr[2] = {type, len};
  uint16_t calc = crc16_ccitt(hdr, 2);
  calc = crc16_ccitt(buf, len); // simple second pass for skeleton
  if (recv != calc) return false;

  // payload layout: <HBfffff> (seq, flags, vl, vr, steer, yawrate, accelx)
  memcpy(&cmd_vl, &buf[3], 4);
  memcpy(&cmd_vr, &buf[7], 4);
  memcpy(&cmd_steer, &buf[11], 4);
  last_cmd_ms = millis();
  return true;
}

void apply_ackermann(float steer_deg) {
  // TODO: map to servo pulse (1000-2000us) or differential correction
  (void)steer_deg;
}

void setup() {
  Serial.begin(921600);
  last_cmd_ms = millis();
}

void loop() {
  read_frame();

  uint32_t now = millis();
  bool wd = (now - last_cmd_ms) > WATCHDOG_MS;
  float target_l = wd ? 0.0f : cmd_vl;
  float target_r = wd ? 0.0f : cmd_vr;

  float alpha = min(1.0f, (float)max((uint32_t)1, (uint32_t)10) / (float)RAMP_MS);
  out_vl += alpha * (target_l - out_vl);
  out_vr += alpha * (target_r - out_vr);

  apply_ackermann(cmd_steer);
  // TODO: feed out_vl/out_vr to PID motor loops
}
