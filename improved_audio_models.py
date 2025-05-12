import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import os
from scipy import signal

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

def extract_dominant_frequencies(audio, n_frequencies=5):
    """Extract the dominant frequencies from the audio signal"""
    # Compute the FFT
    n = len(audio)
    fft_result = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(n, 1/SAMPLE_RATE)
    
    # Get the magnitudes
    magnitudes = np.abs(fft_result)
    
    # Find the indices of the n highest magnitudes
    indices = np.argsort(magnitudes)[-n_frequencies:]
    
    # Get the corresponding frequencies and magnitudes
    dominant_freqs = freqs[indices]
    dominant_mags = magnitudes[indices]
    
    # Normalize magnitudes
    dominant_mags = dominant_mags / np.sum(dominant_mags)
    
    return dominant_freqs, dominant_mags

def improved_rnn_prediction(audio, input_seconds, output_seconds):
    """
    Improved RNN-like prediction that preserves audio structure
    """
    # Get the last part of the audio as input
    input_samples = int(input_seconds * SAMPLE_RATE)
    output_samples = int(output_seconds * SAMPLE_RATE)
    
    input_audio = audio[-input_samples:]
    
    # Extract dominant frequencies and their magnitudes
    dom_freqs, dom_mags = extract_dominant_frequencies(input_audio, n_frequencies=8)
    
    # Generate prediction using a combination of the dominant frequencies
    prediction = np.zeros(output_samples)
    
    # Use the last few samples from the input as initial conditions
    overlap = 1000  # Number of samples to overlap
    prediction[:overlap] = input_audio[-overlap:]
    
    # Generate the rest of the prediction using dominant frequencies
    t = np.arange(output_samples) / SAMPLE_RATE
    
    # Create a base signal using the dominant frequencies
    base_signal = np.zeros(output_samples)
    for i, (freq, mag) in enumerate(zip(dom_freqs, dom_mags)):
        base_signal += mag * np.sin(2 * np.pi * freq * t)
    
    # Apply envelope from the input audio
    envelope = np.abs(librosa.stft(input_audio, n_fft=2048, hop_length=512))
    envelope = np.mean(envelope, axis=0)
    envelope = librosa.util.normalize(envelope)
    
    # Stretch the envelope to match the output length
    envelope = librosa.util.fix_length(envelope, size=output_samples)
    
    # Apply the envelope to the base signal
    base_signal = base_signal * envelope
    
    # Blend the base signal with the initial overlap
    for i in range(overlap, output_samples):
        fade_factor = 0.9  # How much of the previous sample to keep
        prediction[i] = (1 - fade_factor) * base_signal[i] + fade_factor * prediction[i-1]
    
    # Apply a low-pass filter to smooth the prediction
    b, a = signal.butter(3, 0.1)
    prediction = signal.filtfilt(b, a, prediction)
    
    # Normalize the prediction to have the same energy as the input
    input_energy = np.sqrt(np.mean(input_audio**2))
    prediction_energy = np.sqrt(np.mean(prediction**2)) + 1e-10
    prediction = prediction * (input_energy / prediction_energy)
    
    return prediction

def improved_lstm_prediction(audio, input_seconds, output_seconds):
    """
    Improved LSTM-like prediction that preserves audio structure
    """
    # Get the last part of the audio as input
    input_samples = int(input_seconds * SAMPLE_RATE)
    output_samples = int(output_seconds * SAMPLE_RATE)
    
    input_audio = audio[-input_samples:]
    
    # Extract spectral features
    stft = librosa.stft(input_audio, n_fft=2048, hop_length=512)
    mag, phase = librosa.magphase(stft)
    
    # Get the average spectral shape
    avg_spec = np.mean(mag, axis=1)
    
    # Generate prediction
    prediction = np.zeros(output_samples)
    
    # Use the last few samples from the input as initial conditions
    overlap = 2000  # Larger overlap for LSTM-like memory
    prediction[:overlap] = input_audio[-overlap:]
    
    # Extract rhythm information
    onset_env = librosa.onset.onset_strength(y=input_audio, sr=SAMPLE_RATE)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=SAMPLE_RATE)
    
    # Calculate samples per beat
    samples_per_beat = int(60 / tempo * SAMPLE_RATE)
    
    # Generate the rest of the prediction
    for i in range(overlap, output_samples):
        # Memory component (LSTM-like)
        memory_window = 5000  # Longer memory window
        memory_start = max(0, i - memory_window)
        
        # Calculate position in the rhythm cycle
        beat_position = (i % samples_per_beat) / samples_per_beat
        
        # Generate next sample based on memory and rhythm
        if i < 2*overlap:
            # Smooth transition from input
            alpha = (i - overlap) / overlap
            prediction[i] = (1 - alpha) * prediction[i-1] + alpha * (
                0.7 * prediction[i-1] + 
                0.2 * prediction[i-2] + 
                0.1 * np.mean(prediction[memory_start:i])
            )
        else:
            # Full prediction
            prediction[i] = (
                0.6 * prediction[i-1] + 
                0.3 * prediction[i-2] + 
                0.1 * np.mean(prediction[memory_start:i])
            )
            
            # Add rhythmic component
            if beat_position < 0.1:  # Emphasize beats
                prediction[i] *= 1.2
    
    # Apply spectral shaping
    prediction_stft = librosa.stft(prediction, n_fft=2048, hop_length=512)
    prediction_mag, prediction_phase = librosa.magphase(prediction_stft)
    
    # Shape the spectrum to match the input audio's spectral characteristics
    for i in range(prediction_mag.shape[1]):
        prediction_mag[:, i] = prediction_mag[:, i] * avg_spec / (np.mean(prediction_mag[:, i]) + 1e-10)
    
    # Reconstruct audio
    shaped_stft = prediction_mag * prediction_phase
    prediction = librosa.istft(shaped_stft, hop_length=512)
    
    # Ensure the prediction is the right length
    prediction = librosa.util.fix_length(prediction, size=output_samples)
    
    # Normalize the prediction to have the same energy as the input
    input_energy = np.sqrt(np.mean(input_audio**2))
    prediction_energy = np.sqrt(np.mean(prediction**2)) + 1e-10
    prediction = prediction * (input_energy / prediction_energy)
    
    return prediction

def improved_gru_prediction(audio, input_seconds, output_seconds):
    """
    Improved GRU-like prediction that preserves audio structure
    """
    # Get the last part of the audio as input
    input_samples = int(input_seconds * SAMPLE_RATE)
    output_samples = int(output_seconds * SAMPLE_RATE)
    
    input_audio = audio[-input_samples:]
    
    # Extract harmonic and percussive components
    harmonic, percussive = librosa.effects.hpss(input_audio)
    
    # Extract dominant frequencies from harmonic component
    dom_freqs, dom_mags = extract_dominant_frequencies(harmonic, n_frequencies=10)
    
    # Extract rhythm information from percussive component
    onset_env = librosa.onset.onset_strength(y=percussive, sr=SAMPLE_RATE)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=SAMPLE_RATE)
    
    # Calculate samples per beat
    samples_per_beat = int(60 / tempo * SAMPLE_RATE)
    
    # Generate prediction
    prediction = np.zeros(output_samples)
    
    # Use the last few samples from the input as initial conditions
    overlap = 1500  # Medium overlap for GRU-like behavior
    prediction[:overlap] = input_audio[-overlap:]
    
    # Generate harmonic component
    t = np.arange(output_samples) / SAMPLE_RATE
    harmonic_component = np.zeros(output_samples)
    
    for i, (freq, mag) in enumerate(zip(dom_freqs, dom_mags)):
        harmonic_component += mag * np.sin(2 * np.pi * freq * t)
    
    # Generate percussive component based on detected rhythm
    percussive_component = np.zeros(output_samples)
    
    # Create beat pattern
    for i in range(int(output_samples / samples_per_beat) + 1):
        beat_pos = int(i * samples_per_beat)
        if beat_pos < output_samples:
            # Create a decaying envelope for each beat
            decay = np.exp(-np.arange(samples_per_beat) / (samples_per_beat/4))
            end_pos = min(beat_pos + len(decay), output_samples)
            percussive_component[beat_pos:end_pos] += decay[:end_pos-beat_pos]
    
    # Normalize components
    harmonic_component = librosa.util.normalize(harmonic_component)
    percussive_component = librosa.util.normalize(percussive_component)
    
    # Blend components with adaptive weights (GRU-like update gates)
    for i in range(overlap, output_samples):
        # Position in the sequence affects the update gate
        position_factor = i / output_samples
        
        # Update gate: balance between harmonic and percussive components
        update_gate = 0.7 - 0.4 * position_factor
        
        # Reset gate: how much to forget previous prediction
        reset_gate = 0.3 + 0.4 * position_factor
        
        # Calculate new candidate value
        candidate = 0.6 * harmonic_component[i] + 0.4 * percussive_component[i]
        
        # Apply GRU-like update
        prediction[i] = (1 - update_gate) * prediction[i-1] + update_gate * (
            reset_gate * candidate + (1 - reset_gate) * prediction[i-1]
        )
    
    # Apply a bandpass filter to focus on the most important frequency range
    sos = signal.butter(6, [50, 4000], 'bandpass', fs=SAMPLE_RATE, output='sos')
    prediction = signal.sosfilt(sos, prediction)
    
    # Normalize the prediction to have the same energy as the input
    input_energy = np.sqrt(np.mean(input_audio**2))
    prediction_energy = np.sqrt(np.mean(prediction**2)) + 1e-10
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
    plt.title('Improved RNN Prediction')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    
    # Plot LSTM prediction
    plt.subplot(4, 1, 3)
    plt.plot(lstm_pred)
    plt.title('Improved LSTM Prediction')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    
    # Plot GRU prediction
    plt.subplot(4, 1, 4)
    plt.plot(gru_pred)
    plt.title('Improved GRU Prediction')
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
    
    # Generate predictions using improved models
    print("Generating improved RNN prediction...")
    rnn_prediction = improved_rnn_prediction(audio, input_seconds, OUTPUT_SECONDS)
    
    print("Generating improved LSTM prediction...")
    lstm_prediction = improved_lstm_prediction(audio, input_seconds, OUTPUT_SECONDS)
    
    print("Generating improved GRU prediction...")
    gru_prediction = improved_gru_prediction(audio, input_seconds, OUTPUT_SECONDS)
    
    # Save predictions as audio files
    sf.write('results/improved_rnn_prediction.mp3', rnn_prediction, SAMPLE_RATE)
    sf.write('results/improved_lstm_prediction.mp3', lstm_prediction, SAMPLE_RATE)
    sf.write('results/improved_gru_prediction.mp3', gru_prediction, SAMPLE_RATE)
    
    # Get the original next segment for comparison
    original_next = audio[-OUTPUT_SECONDS * SAMPLE_RATE:]
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Plot waveform comparisons
    plot_waveform(original_next, 'Original Next 5 Seconds', 'results/original_next_waveform.png')
    plot_waveform(rnn_prediction, 'Improved RNN Prediction', 'results/improved_rnn_waveform.png')
    plot_waveform(lstm_prediction, 'Improved LSTM Prediction', 'results/improved_lstm_waveform.png')
    plot_waveform(gru_prediction, 'Improved GRU Prediction', 'results/improved_gru_waveform.png')
    
    # Plot spectrogram comparisons
    plot_spectrogram(original_next, 'Original Next 5 Seconds', 'results/original_next_spectrogram.png')
    plot_spectrogram(rnn_prediction, 'Improved RNN Prediction', 'results/improved_rnn_spectrogram.png')
    plot_spectrogram(lstm_prediction, 'Improved LSTM Prediction', 'results/improved_lstm_spectrogram.png')
    plot_spectrogram(gru_prediction, 'Improved GRU Prediction', 'results/improved_gru_spectrogram.png')
    
    # Plot all waveforms together for comparison
    plot_comparison(original_next, rnn_prediction, lstm_prediction, gru_prediction, 
                   'Waveform Comparison', 'results/improved_waveform_comparison.png')
    
    # Calculate MSE for each model
    rnn_mse = np.mean((original_next[:len(rnn_prediction)] - rnn_prediction[:len(original_next)])**2)
    lstm_mse = np.mean((original_next[:len(lstm_prediction)] - lstm_prediction[:len(original_next)])**2)
    gru_mse = np.mean((original_next[:len(gru_prediction)] - gru_prediction[:len(original_next)])**2)
    
    print(f"Improved RNN MSE: {rnn_mse:.6f}")
    print(f"Improved LSTM MSE: {lstm_mse:.6f}")
    print(f"Improved GRU MSE: {gru_mse:.6f}")
    
    # Create a bar chart comparing MSE values
    plt.figure(figsize=(10, 6))
    models = ['Improved RNN', 'Improved LSTM', 'Improved GRU']
    mse_values = [rnn_mse, lstm_mse, gru_mse]
    
    plt.bar(models, mse_values)
    plt.title('MSE Comparison')
    plt.xlabel('Model')
    plt.ylabel('Mean Squared Error')
    plt.tight_layout()
    plt.savefig('results/improved_mse_comparison.png')
    plt.close()
    
    print("All visualizations saved to results directory")

if __name__ == "__main__":
    main()
