from ultralytics import YOLO
import cv2
import time
import numpy as np
import serial

url = "https://huggingface.co/kendrickfff/waste-classification-yolov8-ken/resolve/main/yolov8n-waste-12cls-best.pt"
model = YOLO(url)
TARGET_NAME = "cardboard"

model.predict(source=0, show=True, conf=0.25)
CONF_THRES = 0.25

# 움직임(픽셀) 기준
MOVE_PX_THRES = 35          # 민감하게: 20~30 / 둔하게: 45~70 확인하면서 튜닝
MOVE_HOLD_SEC = 0.25        # 튐 방지

# 쓰러짐 기준: ratio(w/h)
FALL_RATIO_THRES = 1.25     # w/h가 이 이상이면 넘어진거라고 판단 (박스 크기따라서 조정 필요)
RATIO_JUMP_THRES = 0.60     # ratio가 갑자기 변하면 쓰러짐 후보
FALL_HOLD_SEC = 0.20        #오탐 방지하려고 넣음

# UART
SERIAL_PORT = "COM3"        # 포트 맞는 걸로 수정
BAUD = 115200
SEND_COOLDOWN_SEC = 1.0     # 1을 한번 보내면 1초간 재전송 방지 실험해가면서 튜닝 필요


def main():
    # class name -> id 찾기
    names = model.names  # {id: name}
    target_id = None
    for k, v in names.items():
        if v == TARGET_NAME:
            target_id = int(k)
            break
    if target_id is None:
        raise RuntimeError(f"'{TARGET_NAME}' 클래스 없음. 사용 가능: {list(names.values())}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("웹캠 열기 실패")

    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.1)

    prev_center = None
    prev_ratio = None
    move_start = None
    fall_start = None
    last_send_time = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # YOLO 추론 (프레임 단위로 함)
        res = model.predict(frame, conf=CONF_THRES, verbose=False)[0]

        triggered = False
        reason = ""

        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()         # (N,4)
            cls = res.boxes.cls.cpu().numpy().astype(int)
            conf = res.boxes.conf.cpu().numpy()

            # cardboard만 남기기
            mask = (cls == target_id)
            if np.any(mask):
                c_xyxy = xyxy[mask]
                c_conf = conf[mask]

                # 여러 개면 conf 가장 높은 1개 선택
                idx = int(np.argmax(c_conf))
                box = c_xyxy[idx]
                c = float(c_conf[idx])

                x1, y1, x2, y2 = box.astype(int)
                w = max(1, x2 - x1)
                h = max(1, y2 - y1)

                center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
                ratio = w / float(h)
                now = time.time()

                # 화면 표시
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{TARGET_NAME} {c:.2f} r={ratio:.2f}",
                            (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) #디버그 용

                # 많이 움직임: 중심점 이동 거리
                if prev_center is not None:
                    dist = float(np.linalg.norm(center - prev_center))
                    cv2.putText(frame, f"move_px={dist:.1f}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) #디버그용

                    if dist >= MOVE_PX_THRES:
                        if move_start is None:
                            move_start = now
                        if (now - move_start) >= MOVE_HOLD_SEC:
                            triggered = True
                            reason = "MOVE"
                    else:
                        move_start = None
                else:
                    move_start = None

                # 쓰러짐: ratio가 크거나 ratio가 급변
                ratio_event = False
                if ratio >= FALL_RATIO_THRES:
                    ratio_event = True
                if prev_ratio is not None and abs(ratio - prev_ratio) >= RATIO_JUMP_THRES:
                    ratio_event = True

                if ratio_event:
                    if fall_start is None:
                        fall_start = now
                    if (now - fall_start) >= FALL_HOLD_SEC:
                        triggered = True
                        reason = reason or "FALL"
                else:
                    fall_start = None

                prev_center = center
                prev_ratio = ratio

            else:
                # cardboard 미탐지 -> 리셋
                prev_center = None
                prev_ratio = None
                move_start = None
                fall_start = None
        else:
            prev_center = None
            prev_ratio = None
            move_start = None
            fall_start = None

        # UART 전송
        now = time.time()
        if triggered and (now - last_send_time) >= SEND_COOLDOWN_SEC:
            ser.write(b"1\n")
            last_send_time = now
        else:
            ser.write(b"0\n") #평소 상태일 때는 0 보냄

        if triggered:
            cv2.putText(frame, f"TRIGGER: {reason}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3) #디버그 용

        cv2.imshow("cardboard monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    ser.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
