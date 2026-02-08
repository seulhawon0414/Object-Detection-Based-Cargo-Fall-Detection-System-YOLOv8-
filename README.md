# 📦 Object Detection–Based Cargo Fall Detection System (YOLOv8)

## Overview
This project implements a **real-time cargo fall detection system** based on **YOLOv8 object detection**.  
The system detects a specific cargo class (`cardboard`), analyzes its **movement and orientation changes over time**, and triggers an **event signal via UART** when a fall or significant movement is detected.

This project demonstrates how object detection combined with **temporal post-processing** can be applied to practical safety monitoring systems.

---

## Key Features
- Real-time object detection using **YOLOv8**
- Target-specific tracking (`cardboard`)
- Movement detection based on bounding box center displacement
- Fall detection based on bounding box aspect ratio change
- Temporal filtering to reduce false positives
- Event-triggered **UART communication with MCU**
- Optional real-time visualization for debugging

---

## System Architecture

Camera → YOLOv8 Detection → Temporal Analysis (MOVE / FALL)
       → Event Decision → UART Signal → MCU
---
## UART Signaling
- `1` is sent to the MCU when a MOVE or FALL event is confirmed.
- A cooldown mechanism prevents repeated signals from being sent too frequently.
- `0` can optionally represent a normal (non-event) state.

## Requirements
- Python 3.9 or later
- OpenCV
- Ultralytics (YOLOv8)
- NumPy
- pySerial
- USB-connected MCU (optional)


