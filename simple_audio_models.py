import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import os
from scipy.signal import lfilter

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

def simple_rnn_prediction(audio, input_seconds, output_seconds):
    """
    Simple RNN-like prediction using autoregressive linear prediction
    """
    # Get the last part of the audio as input
    input_samples = int(input_seconds * SAMPLE_RATE)
    output_samples = int(output_seconds * SAMPLE_RATE)
    
    input_audio = audio[-input_samples:]
    
    # Use a simpler approach to avoid numerical issues
    order = 10  # Smaller order for stability
    
    # Generate the prediction
    prediction = np.zeros(output_samples)
    
    # Use the last 'order' samples from the input as initial conditions
    for i in range(min(order, output_samples)):
        prediction[i] = input_audio[-order+i]
    
    # Simple decay model (like a very basic RNN)
    if output_samples > order:
        decay_factor = 0.95
        for i in range(order, output_samples):
            # Simple autoregressive model with decay
            prediction[i] = decay_factor * prediction[i-1] + (1-decay_factor) * prediction[i-2]
            
            # Add some randomness to simulate prediction
            prediction[i] += np.random.normal(0, 0.01) * np.std(input_audio)
    
    # Normalize the prediction to have the same energy as the input
    input_energy = np.sqrt(np.mean(input_audio**2))
    prediction_energy = np.sqrt(np.mean(prediction**2)) + 1e-10  # Avoid division by zero
    prediction = prediction * (input_energy / prediction_energy)
    
    return prediction

def simple_lstm_prediction(audio, input_seconds, output_seconds):
    """
    Simple LSTM-like prediction using a more complex autoregressive model
    """
    # Get the last part of the audio as input
    input_samples = int(input_seconds * SAMPLE_RATE)
    output_samples = int(output_seconds * SAMPLE_RATE)
    
    input_audio = audio[-input_samples:]
    
    # Generate the prediction
    prediction = np.zeros(output_samples)
    
    # Use the last few samples from the input as initial conditions
    memory_size = 20  # LSTM-like memory size
    for i in range(min(memory_size, output_samples)):
        prediction[i] = input_audio[-memory_size+i]
    
    # Simple LSTM-like model with memory cells
    if output_samples > memory_size:
        # Initialize memory cells
        memory = np.mean(input_audio[-100:])
        
        for i in range(memory_size, output_samples):
            # Update memory with a forget gate-like mechanism
            forget_factor = 0.7
            memory = forget_factor * memory + (1-forget_factor) * prediction[i-1]
            
            # Generate prediction with memory influence
            prediction[i] = 0.6 * prediction[i-1] + 0.3 * prediction[i-2] + 0.1 * memory
            
            # Add some controlled randomness
            prediction[i] += np.random.normal(0, 0.02) * np.std(input_audio)
    
    # Normalize the prediction to have the same energy as the input
    input_energy = np.sqrt(np.mean(input_audio**2))
    prediction_energy = np.sqrt(np.mean(prediction**2)) + 1e-10  # Avoid division by zero
    prediction = prediction * (input_energy / prediction_energy)
    
    return prediction

def simple_gru_prediction(audio, input_seconds, output_seconds):
    """
    Simple GRU-like prediction using a combination of approaches
    """
    # Get the last part of the audio as input
    input_samples = int(input_seconds * SAMPLE_RATE)
    output_samples = int(output_seconds * SAMPLE_RATE)
    
    input_audio = audio[-input_samples:]
    
    # Generate the prediction
    prediction = np.zeros(output_samples)
    
    # Use the last few samples from the input as initial conditions
    context_size = 15  # GRU-like context size
    for i in range(min(context_size, output_samples)):
        prediction[i] = input_audio[-context_size+i]
    
    # Simple GRU-like model with update and reset mechanisms
    if output_samples > context_size:
        # Initialize hidden state
        hidden = np.mean(input_audio[-50:])
        
        for i in range(context_size, output_samples):
            # Simple reset gate (determines how much to forget)
            reset_gate = 0.5 + 0.2 * np.sin(i / 100)  # Oscillating reset gate
            
            # Simple update gate (determines how much to update)
            update_gate = 0.7 + 0.1 * np.cos(i / 80)  # Oscillating update gate
            
            # Update hidden state
            hidden_candidate = 0.5 * prediction[i-1] + 0.3 * prediction[i-2] + 0.2 * prediction[i-3]
            hidden = (1 - update_gate) * hidden + update_gate * (reset_gate * hidden_candidate)
            
            # Generate prediction
            prediction[i] = 0.7 * hidden + 0.3 * prediction[i-1]
            
            # Add some controlled randomness
            prediction[i] += np.random.normal(0, 0.015) * np.std(input_audio)
    
    # Normalize the prediction to have the same energy as the input
    input_energy = np.sqrt(np.mean(input_audio**2))
    prediction_energy = np.sqrt(np.mean(prediction**2)) + 1e-10  # Avoid division by zero
    prediction = prediction * (input_energy / prediction_energy)
    
    return prediction

def plot_comparison(original, rnn_pred, lstm_pred, gru_pred, title, filename):
    plt.figure(figsize=(12, 8))
    plt.suptitle(title, fontsize=16)
    
    # Plot original audio
    plt.subplot(4, 1, 1)
    plt.plot(original)
    plt.title('Original Audio')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    
    # Plot RNN prediction
    plt.subplot(4, 1, 2)
    plt.plot(rnn_pred)
    plt.title('Simple RNN Prediction')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    
    # Plot LSTM prediction
    plt.subplot(4, 1, 3)
    plt.plot(lstm_pred)
    plt.title('Simple LSTM Prediction')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    
    # Plot GRU prediction
    plt.subplot(4, 1, 4)
    plt.plot(gru_pred)
    plt.title('Simple GRU Prediction')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle
    plt.savefig(filename)
    plt.close()
    print(f"Saved comparison plot to {filename}")

def main():
    # Load audio file
    audio_file = "input.mp3"
    audio = preprocess_audio(audio_file)
    
    # Calculate audio duration
    audio_duration = len(audio) / SAMPLE_RATE
    
    # Use the entire audio except the last OUTPUT_SECONDS as input
    input_seconds = audio_duration - OUTPUT_SECONDS
    print(f"Using {input_seconds:.2f} seconds as input context")
    print(f"Predicting {OUTPUT_SECONDS} seconds of audio")
    
    # Generate predictions using simple models
    print("Generating RNN prediction...")
    rnn_prediction = simple_rnn_prediction(audio, input_seconds, OUTPUT_SECONDS)
    
    print("Generating LSTM prediction...")
    lstm_prediction = simple_lstm_prediction(audio, input_seconds, OUTPUT_SECONDS)
    
    print("Generating GRU prediction...")
    gru_prediction = simple_gru_prediction(audio, input_seconds, OUTPUT_SECONDS)
    
    # Save predictions as audio files
    sf.write('results/simple_rnn_prediction.mp3', rnn_prediction, SAMPLE_RATE)
    sf.write('results/simple_lstm_prediction.mp3', lstm_prediction, SAMPLE_RATE)
    sf.write('results/simple_gru_prediction.mp3', gru_prediction, SAMPLE_RATE)
    
    # Get the original next segment for comparison
    original_next = audio[-OUTPUT_SECONDS * SAMPLE_RATE:]
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Plot waveform comparisons
    plot_waveform(original_next, 'Original Next 5 Seconds', 'results/original_next_waveform.png')
    plot_waveform(rnn_prediction, 'Simple RNN Prediction', 'results/simple_rnn_waveform.png')
    plot_waveform(lstm_prediction, 'Simple LSTM Prediction', 'results/simple_lstm_waveform.png')
    plot_waveform(gru_prediction, 'Simple GRU Prediction', 'results/simple_gru_waveform.png')
    
    # Plot spectrogram comparisons
    plot_spectrogram(original_next, 'Original Next 5 Seconds', 'results/original_next_spectrogram.png')
    plot_spectrogram(rnn_prediction, 'Simple RNN Prediction', 'results/simple_rnn_spectrogram.png')
    plot_spectrogram(lstm_prediction, 'Simple LSTM Prediction', 'results/simple_lstm_spectrogram.png')
    plot_spectrogram(gru_prediction, 'Simple GRU Prediction', 'results/simple_gru_spectrogram.png')
    
    # Plot all waveforms together for comparison
    plot_comparison(original_next, rnn_prediction, lstm_prediction, gru_prediction, 
                   'Waveform Comparison', 'results/waveform_comparison.png')
    
    # Calculate MSE for each model
    rnn_mse = np.mean((original_next[:len(rnn_prediction)] - rnn_prediction[:len(original_next)])**2)
    lstm_mse = np.mean((original_next[:len(lstm_prediction)] - lstm_prediction[:len(original_next)])**2)
    gru_mse = np.mean((original_next[:len(gru_prediction)] - gru_prediction[:len(original_next)])**2)
    
    print(f"Simple RNN MSE: {rnn_mse:.6f}")
    print(f"Simple LSTM MSE: {lstm_mse:.6f}")
    print(f"Simple GRU MSE: {gru_mse:.6f}")
    
    # Create a bar chart comparing MSE values
    plt.figure(figsize=(10, 6))
    models = ['Simple RNN', 'Simple LSTM', 'Simple GRU']
    mse_values = [rnn_mse, lstm_mse, gru_mse]
    
    plt.bar(models, mse_values)
    plt.title('MSE Comparison')
    plt.xlabel('Model')
    plt.ylabel('Mean Squared Error')
    plt.tight_layout()
    plt.savefig('results/mse_comparison.png')
    plt.close()
    
    print("All visualizations saved to results directory")

if __name__ == "__main__":
    main()
