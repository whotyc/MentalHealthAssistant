import cv2
import asyncio
import concurrent.futures
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
from kivy.uix.dropdown import DropDown
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock
import threading
import speech_recognition as sr
import pyttsx3
from plyer import notification
from datetime import datetime
import requests
from io import BytesIO
from kivy.core.image import Image as CoreImage
import matplotlib.pyplot as plt
import bcrypt
import gettext
import os

# Инициализация
cascade_path = os.path.join(os.path.dirname(__file__), '.venv', 'Lib', 'site-packages', 'cv2', 'data', 'haarcascade_frontalface_default.xml')
if not os.path.exists(cascade_path):
    raise FileNotFoundError(f"Haarcascade file not found at {cascade_path}. Please download it from "
                            "https://github.com/opencv/opencv/tree/master/data/haarcascades and place it "
                            "in the cv2/data directory or specify the correct path.")
detector = FER(cascade_file=cascade_path)
cap = cv2.VideoCapture(0)
conn = sqlite3.connect("mental_health.db", check_same_thread=False)

# Настройка переводов
translations_dir = os.path.join(os.path.dirname(__file__), 'translations')
LANGUAGES = ['ru', 'en', 'es', 'fr', 'de', 'zh']
current_lang = 'ru'
_ = gettext.translation('messages', localedir=translations_dir, languages=[current_lang], fallback=True).ugettext

def set_language(lang):
    global _, current_lang
    current_lang = lang
    _ = gettext.translation('messages', localedir=translations_dir, languages=[lang], fallback=True).ugettext

# Модель
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
psychologist = pipeline("text-generation", model="distilgpt2")

async def analyze_voice_async():
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, analyze_voice)

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

async def save_emotion_async(emotion, stress_level, user_id):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: save_emotion(emotion, stress_level, user_id))

def save_emotion(emotion, stress_level, user_id):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO emotions (emotion, timestamp, stress_level, user_id) VALUES (?, ?, ?, ?)", 
                   (emotion, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), stress_level, user_id))
    conn.commit()

async def save_chat_async(user_id, message, response):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: save_chat(user_id, message, response))

def save_chat(user_id, message, response):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chats (user_id, message, response, timestamp) VALUES (?, ?, ?, ?)", 
                   (user_id, message, response, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()

async def save_diary_async(user_id, content, mood):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: save_diary(user_id, content, mood))

def save_diary(user_id, content, mood):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO diaries (user_id, content, timestamp, mood) VALUES (?, ?, ?, ?)", 
                   (user_id, content, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), mood))
    conn.commit()

def plot_stress(user_id):
    cursor = conn.cursor()
    cursor.execute("SELECT stress_level, timestamp FROM emotions WHERE user_id=? ORDER BY timestamp DESC LIMIT 10", (user_id,))
    data = cursor.fetchall()
    stress_levels = [row[0] for row in data]
    times = [row[1][-8:] for row in data]
    plt.figure(figsize=(5, 3))
    plt.plot(times, stress_levels, marker='o', color='b')
    plt.xticks(rotation=45)
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return CoreImage(BytesIO(buf.read()), ext="png").texture

emotion_history = []
async def video_processing_async(user_id):
    while True:
        ret, frame = await asyncio.to_thread(lambda: cap.read())
        if not ret:
            break
        emotions = await asyncio.to_thread(lambda: detector.detect_emotions(frame))
        stress_level = await analyze_voice_async()
        if emotions:
            emotion_dict = emotions[0]["emotions"]
            top_emotion = max(emotion_dict, key=emotion_dict.get)
            emotion_data = list(emotion_dict.values()) + [stress_level]
            emotion_history.append((emotion_data, top_emotion))

            with torch.no_grad():
                pred = model(torch.tensor(emotion_data, dtype=torch.float32).unsqueeze(0))
                pred_emotion = ["angry", "sad", "happy", "stressed"][torch.argmax(pred).item()]
            if stress_level > 0.7:
                pred_emotion = "stressed"

            if len(emotion_history) > 10:
                await asyncio.to_thread(lambda: optimizer.zero_grad())
                outputs = model(torch.tensor(emotion_history[-2][0], dtype=torch.float32).unsqueeze(0))
                target = torch.tensor([["angry", "sad", "happy", "stressed"].index(emotion_history[-2][1])], dtype=torch.long)
                loss = criterion(outputs, target)
                await asyncio.to_thread(lambda: loss.backward())
                await asyncio.to_thread(lambda: optimizer.step())

            notification.notify(title=_(u'MentalHealthAssistant'), message=f'{_("Emotion")}: {pred_emotion}, {_("Stress")}: {stress_level:.2f}', timeout=10)
            await save_emotion_async(pred_emotion, stress_level, user_id)

        await asyncio.to_thread(lambda: cv2.imshow("Mental Health Assistant", frame))
        if await asyncio.to_thread(lambda: cv2.waitKey(1) & 0xFF == ord('q')):
            break

async def recognize_speech_async():
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, recognize_speech)

def recognize_speech():
    with sr.Microphone() as source:
        print(_(u"Listening..."))
        audio = recognizer.listen(source, timeout=5)
    try:
        text = recognizer.recognize_google(audio, language=f"{current_lang}-RU")
        return text
    except sr.UnknownValueError:
        return _(u"Could not recognize speech.")
    except sr.RequestError:
        return _(u"Service recognition error.")

async def text_to_speech_async(text):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: text_to_speech(text))

def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()

async def check_stress_and_notify_async(user_id):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: check_stress_and_notify(user_id))

def check_stress_and_notify(user_id):
    cursor = conn.cursor()
    cursor.execute("SELECT stress_level FROM emotions WHERE user_id=? ORDER BY timestamp DESC LIMIT 1", (user_id,))
    stress = cursor.fetchone()
    if stress and stress[0] > 0.7:
        notification.notify(title=_(u'MentalHealthAssistant'), message=_(u'Your stress level is high. Try meditation!'), timeout=10)

# Хеширование пароля
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

class LoginScreen(Screen):
    def __init__(self, **kwargs):
        super(LoginScreen, self).__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        self.username = TextInput(hint_text=_(u"Username"), size_hint=(1, 0.2))
        self.password = TextInput(hint_text=_(u"Password"), password=True, size_hint=(1, 0.2))
        login_button = Button(text=_(u"Login"), size_hint=(1, 0.2))
        login_button.bind(on_press=self.login)
        register_button = Button(text=_(u"Register"), size_hint=(1, 0.2))
        register_button.bind(on_press=self.register)
        lang_dropdown = Button(text=_(u"Language"), size_hint=(1, 0.2))
        lang_dropdown.bind(on_release=self.show_lang_dropdown)
        self.lang_drop = DropDown()
        for lang in LANGUAGES:
            btn = Button(text=lang, size_hint_y=None, height=44)
            btn.bind(on_release=lambda btn, l=lang: self.set_language(l))
            self.lang_drop.add_widget(btn)
        lang_dropdown.bind(on_release=self.lang_drop.open)
        layout.add_widget(Label(text=_(u"Login"), font_size=30))
        layout.add_widget(self.username)
        layout.add_widget(self.password)
        layout.add_widget(login_button)
        layout.add_widget(register_button)
        layout.add_widget(lang_dropdown)
        self.add_widget(layout)

    def login(self, instance):
        response = requests.post("http://127.0.0.1:5000/login", data={
            "username": self.username.text,
            "password": self.password.text
        })
        if response.status_code == 200:
            self.manager.token = response.json()["access_token"]
            self.manager.user_id = str(requests.get("http://127.0.0.1:5000/api/emotions", 
                                                    headers={"Authorization": f"Bearer {self.manager.token}"}).json()["emotions"][0]["user_id"])
            self.manager.current = 'main'
        else:
            self.username.text = _(u"Login error")

    def register(self, instance):
        response = requests.post("http://127.0.0.1:5000/register", data={
            "username": self.username.text,
            "password": self.password.text
        })
        if response.status_code == 200:
            self.login(instance)
        else:
            self.username.text = _(u"User already exists")

    def show_lang_dropdown(self, instance):
        self.lang_drop.open(instance)

    def set_language(self, lang):
        set_language(lang)
        self.manager.current_screen.ids.lang_dropdown.text = lang
        self.lang_drop.dismiss()

class MainScreen(Screen):
    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        self.status_label = Label(text=_(u"State: unknown"), size_hint=(1, 0.1))
        self.stress_image = Image(size_hint=(1, 0.3))
        self.history_label = Label(text=_(u"History:"), size_hint=(1, 0.2))
        self.chat_input = TextInput(hint_text=_(u"Message to AI Psychologist"), size_hint=(1, 0.1))
        self.chat_output = Label(text=_(u"Chat:"), size_hint=(1, 0.2))
        self.send_button = Button(text=_(u"Send"), size_hint=(1, 0.1))
        self.voice_button = Button(text=_(u"Voice Input"), size_hint=(1, 0.1))
        self.diary_input = TextInput(hint_text=_(u"Write your thoughts..."), size_hint=(1, 0.1))
        self.mood_dropdown = Button(text=_(u"Mood"), size_hint=(1, 0.1))
        self.save_diary_button = Button(text=_(u"Save Diary"), size_hint=(1, 0.1))
        self.send_button.bind(on_press=self.send_message)
        self.voice_button.bind(on_press=self.voice_input)
        self.save_diary_button.bind(on_press=self.save_diary)

        # Dropdown для настроения
        self.mood_drop = DropDown()
        for mood in ["happy", "sad", "angry", "stressed"]:
            btn = Button(text=_(mood.capitalize()), size_hint_y=None, height=44)
            btn.bind(on_release=lambda btn, m=mood: self.set_mood(m))
            self.mood_drop.add_widget(btn)
        self.mood_dropdown.bind(on_release=self.mood_drop.open)

        self.layout.add_widget(self.status_label)
        self.layout.add_widget(self.stress_image)
        self.layout.add_widget(self.history_label)
        self.layout.add_widget(self.chat_input)
        self.layout.add_widget(self.chat_output)
        self.layout.add_widget(self.send_button)
        self.layout.add_widget(self.voice_button)
        self.layout.add_widget(self.diary_input)
        self.layout.add_widget(self.mood_dropdown)
        self.layout.add_widget(self.save_diary_button)
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
                self.status_label.text = f"{_('State')}: {latest['emotion']}, {_('Stress')}: {latest['stress']:.2f}"
                self.history_label.text = f"{_('History')}:\n" + "\n".join([f"{e['time']}: {e['emotion']}" for e in data[:5]])
                self.stress_image.texture = plot_stress(self.manager.user_id)

        asyncio.run(check_stress_and_notify_async(self.manager.user_id))

    def send_message(self, instance):
        message = self.chat_input.text
        response = requests.post("http://127.0.0.1:5000/api/chat", 
                                 json={"message": message}, 
                                 headers={"Authorization": f"Bearer {self.manager.token}"})
        if response.status_code == 200:
            response_text = response.json()["response"]
            asyncio.run(save_chat_async(self.manager.user_id, message, response_text))
            self.chat_output.text = f"{_('Chat')}:\n" + "\n".join([f"{_('You')}: {row[0]}\n{_('AI')}: {row[1]}" for row in cursor.fetchall() if row[0] or row[1]])
        self.chat_input.text = ""

    def voice_input(self, instance):
        text = asyncio.run(recognize_speech_async())
        if text not in [_("Could not recognize speech."), _("Service recognition error.")]:
            response = requests.post("http://127.0.0.1:5000/api/chat", 
                                     json={"voice": text}, 
                                     headers={"Authorization": f"Bearer {self.manager.token}"})
            if response.status_code == 200:
                response_text = response.json()["response"]
                asyncio.run(save_chat_async(self.manager.user_id, text, response_text))
                self.chat_output.text = f"{_('Chat')}:\n" + "\n".join([f"{_('You')}: {row[0]}\n{_('AI')}: {row[1]}" for row in cursor.fetchall() if row[0] or row[1]])
                asyncio.run(text_to_speech_async(response_text))

    def set_mood(self, mood):
        self.mood_dropdown.text = _(mood.capitalize())
        self.mood_drop.dismiss()

    def save_diary(self, instance):
        content = self.diary_input.text
        mood = self.mood_dropdown.text.lower()
        response = requests.post("http://127.0.0.1:5000/api/diary", 
                                 json={"content": content, "mood": mood}, 
                                 headers={"Authorization": f"Bearer {self.manager.token}"})
        if response.status_code == 200:
            self.diary_input.text = ""
            notification.notify(title=_(u'MentalHealthAssistant'), message=_(u'Diary saved!'), timeout=5)

class MentalHealthApp(App):
    def build(self):
        self.sm = ScreenManager()
        self.sm.add_widget(LoginScreen(name='login'))
        self.sm.add_widget(MainScreen(name='main'))
        self.user_id = None
        return self.sm

    def on_start(self):
        set_language('ru')  # Устанавливаем русский по умолчанию

if __name__ == "__main__":
    user_id = "1"  # Заменится после входа
    video_thread = threading.Thread(target=lambda: asyncio.run(video_processing_async(user_id)))
    video_thread.start()
    MentalHealthApp().run()
    cap.release()
    cv2.destroyAllWindows()
    conn.close()