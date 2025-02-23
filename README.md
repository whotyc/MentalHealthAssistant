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
   git clone https://github.com/ВАШ_ЛОГИН/MentalHealthAssistant.git
cd MentalHealthAssistant
