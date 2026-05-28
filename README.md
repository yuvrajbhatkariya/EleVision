# 🐘 EleVision — AI & IoT Powered Elephant-Human Conflict Prevention System

> **Smart India Hackathon 2025** | Problem Statement ID: SIH25167 | Team ID: 75267  
> Theme: Smart Automation | Category: Hardware

[![Live Dashboard](https://img.shields.io/badge/Live%20Dashboard-ele--vision.vercel.app-green)](https://ele-vision.vercel.app/)
[![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-blue)](https://ultralytics.com)
[![Raspberry Pi](https://img.shields.io/badge/Hardware-Raspberry%20Pi%205-red)](https://www.raspberrypi.org/)

---

## 📌 About the Project  [![PPT]](https://www.raspberrypi.org/)

EleVision is a real-time **AI + IoT system** that detects elephant movement near forest-village boundaries and instantly alerts villagers and forest officers — preventing dangerous human-elephant encounters before they happen.

Built for **Chhattisgarh State**, the system uses edge AI on Raspberry Pi, thermal and acoustic sensors, GSM-based SMS alerts, and a live web dashboard to protect both communities and wildlife.

---

## 🚨 Problem

Human-Elephant Conflict (HEC) in India causes:
- Loss of human lives
- Widespread crop and property destruction
- Injury and death to elephants

Most affected areas lack any early warning infrastructure, especially in remote forest zones with no reliable internet.

---

## ✅ Our Solution

| Feature | Description |
|---|---|
| 🔍 **AI Detection** | YOLOv8 model detects elephants in real time from camera frames |
| ⚡ **Edge Processing** | Raspberry Pi 5 runs inference locally — no internet required |
| 📡 **Instant Alerts** | SMS via GSM module, buzzer, and flashlight triggered on detection |
| 🌐 **Live Dashboard** | Web portal for forest officers and villagers with map + alerts |
| 📍 **GPS Tracking** | SIM808 module tracks elephant location |
| 🔮 **Movement Prediction** | Historical data used to predict high-risk zones |

---

## 🏗️ System Architecture

```
PIR Motion Sensor
       ↓
Camera + Audio Activated
       ↓
YOLOv8 Detection (Raspberry Pi 5)
       ↓
Elephant Detected?
    ├── NO  →  Continue Monitoring
    └── YES
          ↓
    Trigger Alerts
    ├── SMS (GSM Module)
    ├── Buzzer + Flashlight
    └── Dashboard Update
          ↓
    Forest Officer & Villager Notified
          ↓
    GPS Location Stored → Movement Prediction
```

---

## 🛠️ Tech Stack

### Hardware
| Component | Purpose |
|---|---|
| Raspberry Pi 5 | Edge AI processing |
| PIR Motion Sensor | Trigger system on movement |
| Thermal Camera | Night / low-light detection |
| Acoustic Sensor | Detect elephant calls |
| SIM808 GSM/GPS Module | SMS alerts + GPS tracking |
| Buzzer & Flashlight | Audio-visual deterrent |
| Solar Battery | Off-grid continuous power |

### Software
| Technology | Purpose |
|---|---|
| Python + YOLOv8 | Elephant detection model |
| OpenCV | Image processing |
| FastAPI | Edge-to-cloud communication |
| Node.js + Express | Backend server |
| React + Vite | Frontend dashboard |
| PostgreSQL | Detection and alert database |
| Firebase Auth | Role-based user authentication |

---

## 📊 Detection Results

The model was trained on a custom elephant dataset and tested across different conditions:

| Test | Detection | Confidence |
|---|---|---|
| Open field (daytime) | ✅ Elephant | 0.93 |
| Mixed animals scene | ✅ Elephant | 0.77 |
| Herd (multiple elephants) | ✅ Elephants | 0.90 |
| Dense vegetation (side view) | ✅ Elephant | 0.64 |

> Detection images from model testing are in the `/docs/results/` folder.

### Sample Detections

<p align="center">
  <img src="docs/results/t2.jpg" width="45%" />
  <img src="docs/results/t3.jpg" width="45%" />
</p>
<p align="center">
  <img src="docs/results/test12.jpg" width="45%" />
  <img src="docs/results/val_batch2_labels.jpg" width="45%" />
</p>

---


## ⚙️ GPIO Pin Connections (Raspberry Pi)

| Component | GPIO Pin |
|---|---|
| PIR Sensor | GPIO 17 |
| PIR LED | GPIO 23 |
| Alert LED | GPIO 27 |
| Buzzer | GPIO 22 |

---


## 📋 requirements.txt

```
ultralytics
opencv-python
numpy
fastapi
uvicorn
requests
python-dotenv
pyserial
RPi.GPIO
psycopg2-binary
twilio
firebase-admin
```

---

## 🌐 Live Dashboard

🔗 **[https://ele-vision.vercel.app/](https://ele-vision.vercel.app/)**

Dashboard features:
- Real-time elephant detection feed
- Village and district-wise alert cards
- Map with high-risk zone visualization
- Alert history and SMS log
- Role-based login (Admin / Forest Officer / Villager)

---

## 📚 Dataset Sources

- Elephant images: [Kaggle – Animal Species Classification](https://www.kaggle.com/code/tobaadesugba/animal-species-classification-using-cnn)
- Thermal images: [Roboflow – Elephant Thermal Dataset](https://universe.roboflow.com/projectbackups/elephant-thermal-bk3tk)
- Elephant sounds: [Pixabay Sound Effects](https://pixabay.com/soundeffects/search/)

---

## 🔭 Future Enhancements

- [ ] Drone-based aerial monitoring
- [ ] WhatsApp alert integration
- [ ] LSTM-based movement prediction
- [ ] Mobile application (Android/iOS)
- [ ] Satellite map integration
- [ ] Multi-animal detection (tigers, bears)

---

## 👥 Team EleVision

**Smart India Hackathon 2025**  
Team ID: 75267 | PS ID: SIH25167  
Theme: Smart Automation | Category: Hardware

---