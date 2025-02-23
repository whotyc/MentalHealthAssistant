import cv2
from fer import FER
import librosa
import numpy as np
import sqlite3
import sounddevice as sd
import torch
import torch.nn as nn
from transformers import pipeline
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock
import threading
import requests
from io import BytesIO
from kivy.core.image import Image as CoreImage
import json

detector = FER()
cap = cv2.VideoCapture(0)
conn = sqlite3.connect("mental_health.db", check_same_thread=False)

class EmotionPredictor(nn.Module):
    def __init__(self):
        super(EmotionPredictor, self).__init__()
        self.fc1 = nn.Linear(8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

model = EmotionPredictor()
psychologist = pipeline("text-generation", model="distilgpt2")

def analyze_voice():
    duration = 2
    fs = 44100
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    audio = recording.flatten()
    mfcc = librosa.feature.mfcc(y=audio, sr=fs, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=audio, sr=fs)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=fs)
    return min(abs(np.mean(mfcc) + np.mean(chroma) + np.mean(spectral_contrast)) / 200, 1.0)

def save_emotion(emotion, stress_level, user_id):
    from datetime import datetime
    conn.execute("INSERT INTO emotions (emotion, timestamp, stress_level, user_id) VALUES (?, ?, ?, ?)", 
                 (emotion, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), stress_level, user_id))
    conn.commit()

def save_chat(user_id, message, response):
    from datetime import datetime
    conn.execute("INSERT INTO chats (user_id, message, response, timestamp) VALUES (?, ?, ?, ?)", 
                 (user_id, message, response, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()

emotion_history = []
def video_processing(user_id):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        emotions = detector.detect_emotions(frame)
        stress_level = analyze_voice()
        if emotions:
            emotion_dict = emotions[0]["emotions"]
            top_emotion = max(emotion_dict, key=emotion_dict.get)
            emotion_data = list(emotion_dict.values()) + [stress_level]
            with torch.no_grad():
                pred = model(torch.tensor(emotion_data, dtype=torch.float32).unsqueeze(0))
                pred_emotion = ["angry", "sad", "happy", "stressed"][torch.argmax(pred).item()]
            if stress_level > 0.7:
                pred_emotion = "stressed"
            save_emotion(pred_emotion, stress_level, user_id)
        cv2.imshow("Mental Health Assistant", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

class LoginScreen(Screen):
    def __init__(self, **kwargs):
        super(LoginScreen, self).__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        self.username = TextInput(hint_text="Логин", size_hint=(1, 0.2))
        self.password = TextInput(hint_text="Пароль", password=True, size_hint=(1, 0.2))
        login_button = Button(text="Войти", size_hint=(1, 0.2))
        login_button.bind(on_press=self.login)
        register_button = Button(text="Регистрация", size_hint=(1, 0.2))
        register_button.bind(on_press=self.register)
        layout.add_widget(Label(text="Вход", font_size=30))
        layout.add_widget(self.username)
        layout.add_widget(self.password)
        layout.add_widget(login_button)
        layout.add_widget(register_button)
        self.add_widget(layout)

    def login(self, instance):
        response = requests.post("http://127.0.0.1:5000/login", data={
            "username": self.username.text,
            "password": self.password.text
        })
        if response.status_code == 200:
            self.manager.token = response.json()["access_token"]
            self.manager.user_id = requests.get("http://127.0.0.1:5000/api/emotions", 
                                                headers={"Authorization": f"Bearer {self.manager.token}"}).json()["emotions"][0]["user_id"]
            self.manager.current = 'main'
        else:
            self.username.text = "Ошибка входа"

    def register(self, instance):
        response = requests.post("http://127.0.0.1:5000/register", data={
            "username": self.username.text,
            "password": self.password.text
        })
        if response.status_code == 200:
            self.login(instance)
        else:
            self.username.text = "Пользователь существует"

class MainScreen(Screen):
    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        self.status_label = Label(text="Состояние: неизвестно", size_hint=(1, 0.1))
        self.stress_image = Image(size_hint=(1, 0.3))
        self.history_label = Label(text="История:", size_hint=(1, 0.2))
        self.chat_input = TextInput(hint_text="Сообщение ИИ-психологу", size_hint=(1, 0.1))
        self.chat_output = Label(text="Чат:", size_hint=(1, 0.2))
        send_button = Button(text="Отправить", size_hint=(1, 0.1))
        send_button.bind(on_press=self.send_message)
        
        self.layout.add_widget(self.status_label)
        self.layout.add_widget(self.stress_image)
        self.layout.add_widget(self.history_label)
        self.layout.add_widget(self.chat_input)
        self.layout.add_widget(self.chat_output)
        self.layout.add_widget(send_button)
        
        self.add_widget(self.layout)
        Clock.schedule_interval(self.update_status, 5.0)

    def update_status(self, dt):
        if not hasattr(self.manager, 'token'):
            return
        response = requests.get("http://127.0.0.1:5000/api/emotions", 
                                headers={"Authorization": f"Bearer {self.manager.token}"})
        if response.status_code == 200:
            data = response.json()["emotions"]
            if data:
                latest = data[0]
                self.status_label.text = f"Состояние: {latest['emotion']}, Стресс: {latest['stress']:.2f}"
                self.history_label.text = "История:\n" + "\n".join([f"{e['time']}: {e['emotion']}" for e in data[:5]])
                
                plt.figure(figsize=(5, 3))
                plt.plot([e['time'][-8:] for e in data], [e['stress'] for e in data], marker='o', color='b')
                plt.xticks(rotation=45)
                buf = BytesIO()
                plt.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                self.stress_image.texture = CoreImage(BytesIO(buf.read()), ext="png").texture
                buf.close()

    def send_message(self, instance):
        message = self.chat_input.text
        response = requests.post("http://127.0.0.1:5000/api/chat", 
                                 json={"message": message}, 
                                 headers={"Authorization": f"Bearer {self.manager.token}"})
        if response.status_code == 200:
            response_text = response.json()["response"]
            save_chat(self.manager.user_id, message, response_text)
            cursor = conn.execute("SELECT message, response FROM chats WHERE user_id=? ORDER BY timestamp DESC LIMIT 5", (self.manager.user_id,))
            self.chat_output.text = "Чат:\n" + "\n".join([f"Вы: {row[0]}\nИИ: {row[1]}" for row in cursor.fetchall()])
        self.chat_input.text = ""

class MentalHealthApp(App):
    def build(self):
        self.sm = ScreenManager()
        self.sm.add_widget(LoginScreen(name='login'))
        self.sm.add_widget(MainScreen(name='main'))
        return self.sm

if __name__ == "__main__":
    user_id = "1"  
    video_thread = threading.Thread(target=video_processing, args=(user_id,))
    video_thread.start()
    MentalHealthApp().run()
    cap.release()
    cv2.destroyAllWindows()
    conn.close()
