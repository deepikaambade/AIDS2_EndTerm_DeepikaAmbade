import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import os

# Constants
SAMPLE_RATE = 16000  # 16kHz
OUTPUT_SECONDS = 5   # Prediction length (fixed at 5 seconds)
FEATURE_DIM = 128    # Mel spectrogram features
HOP_LENGTH = 512     # Hop length for spectrogram

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

def preprocess_audio(file_path, max_duration=None):
    """Load and preprocess audio file, using the entire file if max_duration is None"""
    # For MP3 files, use librosa to load
    print(f"Loading audio file: {file_path}")
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True, duration=max_duration)
    print(f"Audio duration: {len(audio) / SAMPLE_RATE:.2f} seconds")
    return audio

def plot_waveform(audio, title, filename):
    plt.figure(figsize=(10, 4))
    plt.plot(audio)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved waveform plot to {filename}")

def plot_spectrogram(audio, title, filename):
    try:
        # Generate spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=FEATURE_DIM)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spec_db, sr=SAMPLE_RATE, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Saved spectrogram plot to {filename}")
    except Exception as e:
        print(f"Error creating spectrogram visualization: {e}")

def main():
    # Load audio file
    audio_file = "input.mp3"
    audio = preprocess_audio(audio_file)
    
    # Plot waveform
    plot_waveform(audio, 'Full Audio Waveform', 'results/full_audio_waveform.png')
    
    # Plot spectrogram
    plot_spectrogram(audio, 'Full Audio Spectrogram', 'results/full_audio_spectrogram.png')
    
    # Extract the last 5 seconds for visualization
    last_5_seconds = audio[-OUTPUT_SECONDS * SAMPLE_RATE:]
    
    # Plot waveform of the last 5 seconds
    plot_waveform(last_5_seconds, 'Last 5 Seconds Waveform', 'results/last_5_seconds_waveform.png')
    
    # Plot spectrogram of the last 5 seconds
    plot_spectrogram(last_5_seconds, 'Last 5 Seconds Spectrogram', 'results/last_5_seconds_spectrogram.png')
    
    print("Audio visualization complete!")

if __name__ == "__main__":
    main()
