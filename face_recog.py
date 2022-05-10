import cv2
import numpy as np

from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model

import Jetson.GPIO as GPIO
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

SERVO_PIN1 = 32
SERVO_PIN2 = 33
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PIN1, GPIO.OUT)
GPIO.setup(SERVO_PIN2, GPIO.OUT)

# 서보핀 PWM 50Hz로 설정 후 pwm 시작(시작 duty = 0)
servo1 = GPIO.PWM(SERVO_PIN1, 50) # 서보핀 PWM 50Hz로 설정
servo2 = GPIO.PWM(SERVO_PIN2, 50)

servo1.start(0)
servo2.start(0)

# 서보모터 위치 제어 각도를 입력하면 duty값으로 변환 후 리턴하는 함수
def ServoDuty(degree):
    #우리가 가진 모터는 0~180도 동작
    if degree > 180:
        degree = 180
    elif degree < 0:
        degree = 0
    
    # 입력받은 각도를 duty로 변경
    duty = SERVO_MIN_DUTY+(degree*(SERVO_MAX_DUTY-SERVO_MIN_DUTY)/180.0)

    return duty

# 서보모터 1, 2번 제어용 duty값 세팅
Duty1 = [0, 60.0, 120.0, 180.0]
DutyDefault1 = 0
Duty2 = [0, 60.0, 120,0, 180.0]
DutyDefault2 = 0

def draw_ped(img, label, x0, y0, xt, yt, color=(255,127,0), text_color=(255,255,255)):

    (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img,
                  (x0, y0 + baseline),  
                  (max(xt, x0 + w), yt), 
                  color, 
                  2)
    cv2.rectangle(img,
                  (x0, y0 - h),  
                  (x0 + w, y0 + baseline), 
                  color, 
                  -1)  
    cv2.putText(img, 
                label, 
                (x0, y0),                   
                cv2.FONT_HERSHEY_SIMPLEX,     
                0.5,                          
                text_color,                
                1,
                cv2.LINE_AA) 
    return img


# --------- load Haar Cascade model -------------
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# --------- load Keras CNN model -------------
model = load_model("model-cnn-facerecognition.h5")
print("[INFO] finish load model...")

names = np.array(['Ariel_Sharon', 'Colin_Powell', 'Donald_Rumsfeld', 'George_W_Bush', 'Gerhard_Schroeder', 'Hugo_Chavez', 'Jacques_Chirac',
         'Jean_Chretien', 'Jinwoo_Yoon', 'John_Ashcroft', 'Junichiro_Koizumi', 'KeonHee_Jeong', 'KyungSoo_Jeong', 'Serena_Williams', 'Tony_Blair'])

#  
le  = LabelEncoder()
le.fit(names)
labels = le.classes_
name_vec = le.transform(names)

cap = cv2.VideoCapture(0)
while cap.isOpened() :
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (50, 50))
            face_img = face_img.reshape(1, 50, 50, 1)
            
            result = model.predict(face_img)
            idx = result.argmax(axis=1)
            confidence = result.max(axis=1)*100
            label_text = "%s (%.2f %%)" % (names[idx], confidence)

            if confidence > 90:
                id = labels[idx]
                label_text = "%s (%.2f %%)" % (labels[idx], confidence)
                if id == 'KeonHee_Jeong' :
                    DutyDefault1 = DutyDefault1 + 1
                    servo1.ChangeDutyCycle(ServoDuty(Duty1[DutyDefault1]))
                    if(DutyDefault1 == 3) :
                        DutyDefault1 = 0
                if id == 'Jinwoo_Yoon' :
                    DutyDefault2 = DutyDefault2 + 1
                    servo2.ChangeDutyCycle(ServoDuty(Duty2[DutyDefault2]))
                    if(DutyDefault2 == 3) :
                        DutyDefault2 = 0
            else :
                label_text = "N/A" #continue로 써도 됨
            frame = draw_ped(frame, label_text, x, y, x + w, y + h, color=(0,255,255), text_color=(50,50,50))
       
        cv2.imshow('Detect Face', frame)
        time.sleep(0.5)
    else :
        break
    if cv2.waitKey(10) == ord('q'):
        break
        
cv2.destroyAllWindows()
cap.release()