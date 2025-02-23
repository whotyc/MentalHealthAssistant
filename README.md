# MentalHealthAssistant

MentalHealthAssistant is an innovative Python application designed to monitor mental health in real time. It analyzes emotions through a webcam and stress levels through a microphone, offers personalized recommendations (including Spotify playlists) and provides a chat with an AI psychologist. The project includes a web version on Flask and a mobile application on Kivy, with support for authorization, API and data visualization.

## Main functions
- **Emotion Analysis**: Uses FER to recognize emotions from a webcam.
- **Stress Assessment**: Analyzes voice through MFCC, chromatic features and spectral contrast.
- **Personalization**: The neural network (PyTorch) adapts to the user.
- **Recommendations**: Integration with Spotify for music playlists.
- **AI psychologist**: Chat based on `distilgpt2` for support.
- **Web version**: Flask with JWT authorization, stress visualization and modern UI.
- **Mobile application**: Kivy with synchronization via API and improved interface.

## Requirements
- Python 3.8+
- Webcam and microphone
- Spotify API keys (see below)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/whotyc/MentalHealthAssistant.git

## Install dependencies:
   ```bash
   pip install opencv-python fer librosa sounddevice flask flask-jwt-extended numpy torch transformers spotipy kivy matplotlib requests aiohttp

##Configure the Spotify API:
Sign up for Spotify Developer.
Create an application and get the client_id and client_secret.
Paste them into both files (web_version.py and mobile_version.py ), replacing YOUR_SPOTIFY_CLIENT_ID and YOUR_SPOTIFY_CLIENT_SECRET.
Using
Launching the web version:
bash
Transfer
Copy
python web_version.py
Open up http://127.0.0.1:5000/login in the browser.
Register or log in.
Launching the mobile app:
bash
Transfer
Copy
python mobile_version.py
Enter your username and password to synchronize with the web version.
Functionality:
Video: Analyzes emotions and stress, displays recommendations.
Web: Two tabs — history (with stress graph) and chat with AI.
Mobile: Login screen and main screen with status, history and chat.
Project
structure web_version.py : A web application on Flask with an API.
mobile_version.py : Mobile app on Kivy.
mental_health.db: SQLite database for users, emotions, and chats.
Technical details
Security: JWT for authorization, session for username.
Performance: Asynchrony with asyncio in the web version.
AI: distilgpt2 for psychologist, PyTorch for personalization.
UI: CSS for the web, ScreenManager for Kivy.
Analysis: OpenCV (FER), librosa for audio, Matplotlib for graphs.
Limitations and improvements
Performance: Streaming video processing; requires asyncio-compatible OpenCV for full asynchrony.
AI: distilgpt2 is basic; to improve, connect an API (for example, xAI) or teach psychological dialogues on the dataset.
UI: Animations are possible (Bootstrap for web, KivyMD for mobile).
License
MIT License. See the LICENSE file (recommended to add).

Contacts
If you have any questions or suggestions, create an issue or write to [YOUR_EMAIL].
