import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader

# Try to import librosa.display for spectrogram visualization
try:
    import librosa.display
except ImportError:
    print("Warning: librosa.display not available, some visualizations may be limited")

# Constants
SAMPLE_RATE = 16000  # 16kHz
OUTPUT_SECONDS = 5   # Prediction length (fixed at 5 seconds)
FEATURE_DIM = 128    # Mel spectrogram features
HOP_LENGTH = 512     # Hop length for spectrogram

# Audio preprocessing
def preprocess_audio(file_path, max_duration=None):
    """Load and preprocess audio file, using the entire file if max_duration is None"""
    # For MP3 files, use librosa to load
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True, duration=max_duration)
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
            y=audio, sr=SAMPLE_RATE, n_mels=FEATURE_DIM, hop_length=512
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

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])

# Training function
def train_model(model, train_loader, epochs=50, lr=0.001):
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

            # Reshape for LSTM input [batch, time_steps, features]
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
    plt.title('LSTM Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')

    plt.savefig('results/lstm_training_loss.png')
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
def plot_waveform_comparison(original, predicted):
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
    plt.title('LSTM Predicted Audio')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.tight_layout()

    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')

    plt.savefig('results/lstm_waveform_comparison.png')
    plt.close()

def plot_spectrogram_comparison(original, predicted):
    try:
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
        plt.title('LSTM Predicted Mel Spectrogram')

        plt.tight_layout()

        # Create results directory if it doesn't exist
        if not os.path.exists('results'):
            os.makedirs('results')

        plt.savefig('results/lstm_spectrogram_comparison.png')
        plt.close()
    except Exception as e:
        print(f"Error creating spectrogram visualization: {e}")

# Main execution
def main():
    # Load audio file (use the entire file)
    audio_file = "input.mp3"
    print(f"Loading audio file: {audio_file}")
    audio = preprocess_audio(audio_file)

    # Calculate audio duration
    audio_duration = len(audio) / SAMPLE_RATE
    print(f"Audio duration: {audio_duration:.2f} seconds")

    # Use the entire audio except the last OUTPUT_SECONDS as input
    input_seconds = audio_duration - OUTPUT_SECONDS
    print(f"Using {input_seconds:.2f} seconds as input context")
    print(f"Predicting {OUTPUT_SECONDS} seconds of audio")

    # Convert seconds to frames
    input_frames = int(input_seconds * SAMPLE_RATE / HOP_LENGTH)
    output_frames = int(OUTPUT_SECONDS * SAMPLE_RATE / HOP_LENGTH)

    # Create dataset and dataloader
    dataset = AudioPredictionDataset(
        audio,
        input_len=input_frames,
        output_len=output_frames
    )

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model
    input_dim = FEATURE_DIM
    hidden_dim = 256
    output_dim = FEATURE_DIM * output_frames

    lstm_model = LSTMModel(input_dim, hidden_dim, output_dim)

    # Train model
    print("Training LSTM model...")
    train_model(lstm_model, train_loader)

    # Use the entire audio except the last OUTPUT_SECONDS as input for prediction
    input_audio = audio[:-OUTPUT_SECONDS * SAMPLE_RATE]

    # Generate prediction
    print("Generating LSTM prediction...")
    lstm_prediction = generate_prediction(lstm_model, input_audio, input_frames, output_frames)

    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')

    # Save prediction as audio file
    prediction_file = 'results/lstm_prediction.mp3'
    sf.write(prediction_file, lstm_prediction, SAMPLE_RATE)
    print(f"Prediction saved to {prediction_file}")

    # Get the original next segment for comparison
    original_next = audio[-OUTPUT_SECONDS * SAMPLE_RATE:]

    # Calculate MSE
    min_len = min(len(original_next), len(lstm_prediction))
    lstm_mse = np.mean((original_next[:min_len] - lstm_prediction[:min_len])**2)
    print(f"LSTM MSE: {lstm_mse:.6f}")

    # Create visualizations
    print("Creating visualizations...")
    plot_waveform_comparison(original_next, lstm_prediction)
    plot_spectrogram_comparison(original_next, lstm_prediction)
    print("Visualizations saved to results directory")

if __name__ == "__main__":
    main()