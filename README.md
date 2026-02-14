# Unified Robot Stack v3.8

Bu sürümde v3.7 üzerine zaman senkronizasyonu, adaptif fusion güveni ve gerçek-zamanlı kontrol bütçesi iyileştirmeleri eklendi.

## 1) Sade Blok Diyagram (Önerilen)
1. **Perception**: Camera + LiDAR + ScanMatcher
2. **Localization**: EKF (`x=[x,y,yaw,v,gyro_bias]`)
3. **Planning/Control**: Nav2 global path + MPC-lite + DWA local avoidance
4. **Safety Layer (bağımsız)**: Risk + Watchdog + BMS + Manual E-stop birleşimi
5. **Bridge/Actuation**: Pi → ESP32 binary frame + CRC + watchdog contract

## 2) SLAM / TF Görev Ayrımı (Net)
- Tek referans SLAM akışı: `slam_toolbox`.
- **Önerilen TF görev ayrımı**:
  - SLAM: `map -> odom`
  - EKF: `odom -> base_link`
  - Sensör statik: `base_link -> laser`
- `TfChainNode` varsayılan olarak sadece `base_link -> laser` statik ve `odom -> base_link` dinamik yayınlar. `map->odom` yayınını varsayılan kapalı tutar (`publish_map_to_odom=false`).



## 2.1) Daha Gelişmiş Haritalama (SLAM + Nav2 Entegrasyon Görünürlüğü)
- Yeni `SlamNav2BridgeNode` eklendi (`mapbridge` modu).
- `/map` akışının güncelliğini izleyip `/slam/ready`, `/slam/status`, `/heartbeat/slam` yayınlar.
- `Nav2LifecycleGuardNode` artık `/slam/ready` sinyalini de şart koşar; yani plan taze olsa bile harita stale ise `nav2/ready=false` olur.
- Bu sayede SLAM katmanının Nav2 ile entegrasyonu kodda görünür, ölçülebilir ve safety zincirine bağlı hale gelir.

## 3) EKF Modeli (v3.8 Güncelleme)
`EkfOdomNode` durum vektörü:
- `x = [x, y, yaw, v, gyro_bias]`

Model:
- Propagasyon: Jacobian `F` ile
- Süreç kovaryansı: `Q`
- Ölçüm güncellemesi: `H`, `R`
- Ölçümler:
  - Wheel odom `v`
  - Wheel yaw-rate ile IMU bias düzeltmesi
  - GPS (lever-arm correction ile `x,y`)
  - Scan yaw pseudo-measurement

Dinamik güven/ağırlık:
- `Q` hız ve yaw-rate ile adaptif ölçeklenir; GPS dropout süresi arttıkça süreç gürültüsü kontrollü artırılır.
- GPS ölçüm kovaryansı `R_gps`, gelen `position_covariance` + `position_covariance_type` + fix status ile dinamik güncellenir.
- Stale sensör paketleri (IMU/Wheel/GPS) `max_sensor_age_s` eşiği ile filtrelenir.
- EKF `/ekf/fusion_mode` yayınlayarak hangi sensör kümesinin baskın olduğunu görünür yapar.

## 4) MPC/DWA Kontrol Modeli
- MPC-lite: steer sweep ile kısa ufukta path sapmasını minimize eder.
- DWA: dinamik pencere içinde `(v, steer)` adayları tarar:
  - maliyet (local costmap)
  - heading hatası
  - steer yumuşaklığı
  - semantic lateral bias (S manevrası)
- Direksiyon rate ve long accel/decel limitleri korunur.
- MPC çözüm süresi `/perf/mpc_solve_ms` ile izlenir; çözüm zamanı bütçeyi aşarsa hız hedefi otomatik düşürülür.
- Odom/plan verisi stale olursa kontrol katmanı güvenli bekleme moduna geçer.

## 5) Safety Layer (Bağımsız)
`SafetySupervisorNode` girişleri:
- `/emergency_stop` (RiskNode)
- `/failsafe/estop` (Watchdog)
- `/battery/status` (BMS)
- `/manual_estop`

Çıkışlar:
- `/safety/estop`
- `/safety/brake_override`
- `/safety/mode`

Autonomy ve Bridge doğrudan safety çıkışlarını tüketir.

## 6) Ölçülebilir Performans Metrikleri
- `/perf/ekf_loop_dt_ms`
- `/perf/mpc_loop_dt_ms`
- `/perf/summary` (heartbeat age + loop dt + mpc solve özeti)
- `/perf/mpc_solve_ms`

Bu metrikler jitter ve gerçek zamanlılık takibini ölçülebilir hale getirir.

## 7) Fiziksel Teknik Özellikler
- Teker çapı: `0.125 m`
- Teker yarıçapı: `0.0625 m`
- Wheelbase: `0.24 m`
- Track width: `0.302 m`
- Encoder: `64 CPR`, redüktör `30:1`, çıkış `1920 CPR`, quadrature `7680 count/rev`

## 8) Kritik Tuning Notları
- `wheel_base_m=0.24` değeri mekanik aks mesafesi ile birebir ölçülmeli.
- `slip_deadband_speed_mps=0.30` düşük hız false slip için kritik.
- `gps_lever_arm_x_m / gps_lever_arm_y_m` cm hassasiyetinde ölçülmeli.

## 9) Çalıştırma (İşlem Başlığına Göre)
```bash
python3 processes/01_perception.py vision
python3 processes/02_planning_control.py mpc
python3 processes/03_state_safety.py mapbridge
python3 processes/03_state_safety.py ekf
python3 processes/03_state_safety.py safety
python3 processes/03_state_safety.py perf
python3 processes/04_bridge_io.py
python3 bms_monitor_node.py
python3 web_status_bridge.py
```
