import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os

# Try to import librosa.display for spectrogram visualization
try:
    import librosa.display
except ImportError:
    print("Warning: librosa.display not available, some visualizations may be limited")

# Constants
SAMPLE_RATE = 16000  # 16kHz
INPUT_SECONDS = None  # Will be set based on the audio length
OUTPUT_SECONDS = 5    # Prediction length (fixed at 5 seconds)
FEATURE_DIM = 128     # Mel spectrogram features
HOP_LENGTH = 512      # Hop length for spectrogram

# Audio preprocessing
def preprocess_audio(file_path, max_duration=None):
    """Load and preprocess audio file, using the entire file if max_duration is None"""
    # For MP3 files, use librosa to load
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True, duration=max_duration)
    return audio

# Dataset for audio sequence prediction
class AudioPredictionDataset(Dataset):
    def __init__(self, audio, input_len, output_len, hop_len=1):
        self.audio = audio
        self.input_len = input_len
        self.output_len = output_len
        self.hop_len = hop_len

        # Convert to mel spectrogram
        self.mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=SAMPLE_RATE, n_mels=FEATURE_DIM, hop_length=HOP_LENGTH
        )
        self.mel_spec = librosa.power_to_db(self.mel_spec, ref=np.max)

        # Normalize
        self.mel_spec = (self.mel_spec - self.mel_spec.mean()) / (self.mel_spec.std() + 1e-8)

        # Calculate valid sequences
        self.num_frames = self.mel_spec.shape[1]
        self.seq_len = input_len + output_len
        self.valid_indices = range(0, self.num_frames - self.seq_len + 1, hop_len)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.seq_len

        sequence = self.mel_spec[:, start_idx:end_idx]

        # Input and target sequences
        x = sequence[:, :self.input_len]
        y = sequence[:, self.input_len:self.seq_len]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# RNN Models
class VanillaRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output, _ = self.rnn(x)
        return self.fc(output[:, -1, :])

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output, _ = self.gru(x)
        return self.fc(output[:, -1, :])

# Training function
def train_model(model, train_loader, model_name, epochs=50, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Reshape for RNN input [batch, time_steps, features]
            x_batch = x_batch.permute(0, 2, 1)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(x_batch)

            # Reshape target to match output
            y_batch = y_batch.reshape(y_batch.size(0), -1)

            # Calculate loss
            loss = criterion(outputs, y_batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}')

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), losses)
    plt.title(f'{model_name} Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(f'{model_name}_training_loss.png')
    plt.close()

    return losses

# Generate audio from model prediction
def generate_prediction(model, input_audio, input_len, output_len):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Convert to mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=input_audio, sr=SAMPLE_RATE, n_mels=FEATURE_DIM, hop_length=HOP_LENGTH
    )
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize
    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

    # Get input sequence
    input_seq = mel_spec[:, -input_len:]
    input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(device)

    with torch.no_grad():
        # Get prediction
        prediction = model(input_tensor)
        prediction = prediction.cpu().numpy().reshape(FEATURE_DIM, output_len)

    # Denormalize
    prediction = prediction * (mel_spec.std() + 1e-8) + mel_spec.mean()

    # Convert back to audio using Griffin-Lim
    prediction_db = librosa.db_to_power(prediction)
    predicted_audio = librosa.feature.inverse.mel_to_audio(
        prediction_db, sr=SAMPLE_RATE, hop_length=HOP_LENGTH
    )

    return predicted_audio

# Visualization functions
def plot_waveform_comparison(original, predicted, model_name):
    plt.figure(figsize=(15, 6))

    # Plot original audio
    plt.subplot(2, 1, 1)
    plt.plot(original)
    plt.title('Original Audio')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    # Plot predicted audio
    plt.subplot(2, 1, 2)
    plt.plot(predicted)
    plt.title(f'{model_name} Predicted Audio')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.savefig(f'{model_name}_waveform_comparison.png')
    plt.close()

def plot_spectrogram_comparison(original, predicted, model_name):
    # Generate spectrograms
    orig_spec = librosa.feature.melspectrogram(y=original, sr=SAMPLE_RATE, n_mels=FEATURE_DIM)
    orig_spec_db = librosa.power_to_db(orig_spec, ref=np.max)

    pred_spec = librosa.feature.melspectrogram(y=predicted, sr=SAMPLE_RATE, n_mels=FEATURE_DIM)
    pred_spec_db = librosa.power_to_db(pred_spec, ref=np.max)

    plt.figure(figsize=(15, 8))

    # Plot original spectrogram
    plt.subplot(2, 1, 1)
    librosa.display.specshow(orig_spec_db, sr=SAMPLE_RATE, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Original Mel Spectrogram')

    # Plot predicted spectrogram
    plt.subplot(2, 1, 2)
    librosa.display.specshow(pred_spec_db, sr=SAMPLE_RATE, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'{model_name} Predicted Mel Spectrogram')

    plt.tight_layout()
    plt.savefig(f'{model_name}_spectrogram_comparison.png')
    plt.close()

# Create output directory for results
def create_output_dir():
    if not os.path.exists('results'):
        os.makedirs('results')
    return 'results'

# Main function if this file is run directly
def main():
    """
    Simple test function to demonstrate the model's capabilities
    For full functionality, use run_audio_prediction.py
    """
    import argparse

    parser = argparse.ArgumentParser(description='Audio Prediction Model Test')
    parser.add_argument('--audio_file', type=str, default='input.mp3', help='Input audio file path')
    parser.add_argument('--model', type=str, default='lstm', choices=['rnn', 'lstm', 'gru'], help='Model type')
    args = parser.parse_args()

    # Create output directory
    output_dir = create_output_dir()

    # Load audio file
    print(f"Loading audio file: {args.audio_file}")
    audio = preprocess_audio(args.audio_file)
    audio_duration = len(audio) / SAMPLE_RATE
    print(f"Audio duration: {audio_duration:.2f} seconds")

    # Set input and output parameters
    output_seconds = 5
    input_seconds = audio_duration - output_seconds

    # Convert to frames
    input_frames = int(input_seconds * SAMPLE_RATE / HOP_LENGTH)
    output_frames = int(output_seconds * SAMPLE_RATE / HOP_LENGTH)

    # Create dataset
    dataset = AudioPredictionDataset(
        audio,
        input_len=input_frames,
        output_len=output_frames
    )

    # Create dataloader
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model
    input_dim = FEATURE_DIM
    hidden_dim = 256
    output_dim = FEATURE_DIM * output_frames

    # Create model based on user choice
    if args.model == 'rnn':
        model = VanillaRNN(input_dim, hidden_dim, output_dim)
        model_name = 'RNN'
    elif args.model == 'lstm':
        model = LSTMModel(input_dim, hidden_dim, output_dim)
        model_name = 'LSTM'
    else:  # gru
        model = GRUModel(input_dim, hidden_dim, output_dim)
        model_name = 'GRU'

    # Train model
    print(f"Training {model_name} model...")
    train_model(model, train_loader, model_name, epochs=30)

    # Generate prediction
    input_audio = audio[:-output_seconds * SAMPLE_RATE]
    prediction = generate_prediction(model, input_audio, input_frames, output_frames)

    # Save prediction
    prediction_file = os.path.join(output_dir, f'{model_name.lower()}_prediction.mp3')
    sf.write(prediction_file, prediction, SAMPLE_RATE)
    print(f"Prediction saved to {prediction_file}")

    # Create visualizations
    original_next = audio[-output_seconds * SAMPLE_RATE:]
    plot_waveform_comparison(original_next, prediction, model_name)

    try:
        plot_spectrogram_comparison(original_next, prediction, model_name)
    except Exception as e:
        print(f"Error creating spectrogram visualization: {e}")

    # Calculate MSE
    min_len = min(len(original_next), len(prediction))
    mse = np.mean((original_next[:min_len] - prediction[:min_len])**2)
    print(f"{model_name} MSE: {mse:.6f}")

if __name__ == "__main__":
    main()
