import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy.interpolate import interp1d

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

def find_repeating_patterns(audio, min_pattern_length=0.5, max_pattern_length=2.0):
    """Find repeating patterns in the audio signal"""
    # Convert pattern lengths from seconds to samples
    min_samples = int(min_pattern_length * SAMPLE_RATE)
    max_samples = int(max_pattern_length * SAMPLE_RATE)

    # Ensure max_samples is not larger than the audio length
    max_samples = min(max_samples, len(audio) // 2)

    # Use a default pattern length if audio is too short
    if max_samples < min_samples:
        return int(0.5 * SAMPLE_RATE)  # Default 0.5-second pattern

    try:
        # Compute autocorrelation to find repeating patterns
        autocorr = librosa.autocorrelate(audio, max_size=max_samples)

        # Find peaks in autocorrelation (potential pattern lengths)
        peaks = librosa.util.peak_pick(autocorr, pre_max=20, post_max=20, pre_avg=30, post_avg=30, delta=0.5, wait=min_samples)

        # If no peaks found, return a default pattern length
        if len(peaks) == 0:
            return int(1.0 * SAMPLE_RATE)  # Default 1-second pattern

        # Return the most prominent peak (strongest repeating pattern)
        peak_values = [autocorr[p] for p in peaks]
        best_peak_idx = np.argmax(peak_values)
        pattern_length = peaks[best_peak_idx]

        # Ensure pattern length is at least min_samples
        if pattern_length < min_samples:
            pattern_length = min_samples

        return pattern_length
    except Exception as e:
        print(f"Error in pattern detection: {e}")
        return int(1.0 * SAMPLE_RATE)  # Default 1-second pattern

def extract_audio_features(audio):
    """Extract comprehensive features from the audio signal"""
    features = {}

    # Spectral features
    stft = librosa.stft(audio, n_fft=2048, hop_length=512)
    mag, phase = librosa.magphase(stft)
    features['magnitude'] = mag
    features['phase'] = phase

    # Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=FEATURE_DIM)
    features['mel_spec'] = mel_spec

    # Chromagram (pitch content)
    chroma = librosa.feature.chroma_stft(S=np.abs(stft), sr=SAMPLE_RATE)
    features['chroma'] = chroma

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(S=np.abs(stft), sr=SAMPLE_RATE)
    features['contrast'] = contrast

    # Onset strength
    onset_env = librosa.onset.onset_strength(y=audio, sr=SAMPLE_RATE)
    features['onset_env'] = onset_env

    # Tempo and beat
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=SAMPLE_RATE)
    features['tempo'] = tempo
    features['beats'] = beats

    # MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=20)
    features['mfcc'] = mfcc

    # Harmonic and percussive components
    harmonic, percussive = librosa.effects.hpss(audio)
    features['harmonic'] = harmonic
    features['percussive'] = percussive

    # Find repeating patterns
    pattern_length = find_repeating_patterns(audio)
    features['pattern_length'] = pattern_length

    return features

def advanced_rnn_prediction(audio, input_seconds, output_seconds):
    """
    Advanced RNN-like prediction using pattern continuation and spectral modeling
    """
    # Get the last part of the audio as input
    input_samples = int(input_seconds * SAMPLE_RATE)
    output_samples = int(output_seconds * SAMPLE_RATE)

    input_audio = audio[-input_samples:]

    # Extract comprehensive features
    features = extract_audio_features(input_audio)

    # Find the pattern length (in samples)
    pattern_length = features['pattern_length']
    print(f"Detected pattern length: {pattern_length/SAMPLE_RATE:.2f} seconds")

    # Ensure pattern length is reasonable
    if pattern_length < 100:  # If pattern is too short
        pattern_length = int(0.5 * SAMPLE_RATE)  # Use 0.5 seconds as default
        print(f"Using default pattern length: {pattern_length/SAMPLE_RATE:.2f} seconds")

    # Generate prediction by continuing the pattern with variations
    prediction = np.zeros(output_samples)

    # Use a portion of the input as initial condition
    overlap = min(2000, pattern_length)
    prediction[:overlap] = input_audio[-overlap:]

    # Continue with pattern-based prediction
    pos = overlap
    while pos < output_samples:
        # Determine where to sample from in the input audio
        # Use segments from the latter part of the input audio
        available_input = len(input_audio) - pattern_length
        if available_input <= 0:
            # If input is too short, just repeat the last part
            segment_start = 0
        else:
            # Prefer segments from the latter half of the input
            segment_start = max(0, available_input // 2) + np.random.randint(0, available_input // 2)

        # Get a segment from the input
        segment_length = min(pattern_length, output_samples - pos)
        if segment_start + segment_length > len(input_audio):
            segment_start = len(input_audio) - segment_length

        segment = input_audio[segment_start:segment_start + segment_length]

        # Apply some variations to make it sound less repetitive
        variation_factor = 0.05 + 0.1 * (pos / output_samples)  # Increase variation over time
        segment = segment * (1 + variation_factor * np.random.randn(len(segment)))

        # Apply a crossfade at the beginning of the segment for smooth transition
        crossfade_length = min(200, segment_length // 4)
        for i in range(crossfade_length):
            alpha = i / crossfade_length
            if pos + i < output_samples:
                prediction[pos + i] = (1 - alpha) * prediction[pos + i - 1] + alpha * segment[i]

        # Copy the rest of the segment
        if segment_length > crossfade_length:
            prediction[pos + crossfade_length:pos + segment_length] = segment[crossfade_length:]

        pos += segment_length

    # Apply spectral shaping to match the input audio characteristics
    prediction_stft = librosa.stft(prediction, n_fft=2048, hop_length=512)
    prediction_mag, prediction_phase = librosa.magphase(prediction_stft)

    # Get average spectral shape from input
    input_mag = features['magnitude']
    avg_spec_shape = np.mean(input_mag, axis=1)

    # Shape the spectrum
    for i in range(prediction_mag.shape[1]):
        current_shape = np.mean(prediction_mag[:, i]) + 1e-10
        prediction_mag[:, i] = prediction_mag[:, i] * (avg_spec_shape / current_shape)

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

def advanced_lstm_prediction(audio, input_seconds, output_seconds):
    """
    Advanced LSTM-like prediction using long-term memory and context modeling
    """
    # Get the last part of the audio as input
    input_samples = int(input_seconds * SAMPLE_RATE)
    output_samples = int(output_seconds * SAMPLE_RATE)

    input_audio = audio[-input_samples:]

    # Extract comprehensive features
    features = extract_audio_features(input_audio)

    # Get tempo information
    tempo = features.get('tempo', 120)  # Default to 120 BPM if not detected

    # Calculate samples per beat
    samples_per_beat = int(60 / tempo * SAMPLE_RATE)

    # Generate prediction
    prediction = np.zeros(output_samples)

    # Use a portion of the input as initial condition
    overlap = min(3000, samples_per_beat * 2)
    if overlap > len(input_audio):
        overlap = len(input_audio) // 2
    prediction[:overlap] = input_audio[-overlap:]

    # Analyze the structure of the input audio
    # Look for sections with similar characteristics
    section_length = samples_per_beat * 4  # Typical 4-beat phrase
    if section_length < 100:  # Ensure reasonable section length
        section_length = int(0.5 * SAMPLE_RATE)

    num_sections = max(1, len(input_audio) // section_length)

    # Create a similarity matrix between sections
    similarity_matrix = np.zeros((num_sections, num_sections))
    sections = []

    for i in range(num_sections):
        start_i = i * section_length
        end_i = min(start_i + section_length, len(input_audio))
        section_i = input_audio[start_i:end_i]
        if len(section_i) < section_length:
            section_i = np.pad(section_i, (0, section_length - len(section_i)))
        sections.append(section_i)

        for j in range(i+1):
            start_j = j * section_length
            end_j = min(start_j + section_length, len(input_audio))
            section_j = input_audio[start_j:end_j]
            if len(section_j) < section_length:
                section_j = np.pad(section_j, (0, section_length - len(section_j)))

            try:
                # Calculate similarity (correlation)
                correlation = np.corrcoef(section_i, section_j)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.5  # Default value if correlation is NaN
            except:
                correlation = 0.5  # Default value if correlation calculation fails

            similarity_matrix[i, j] = correlation
            similarity_matrix[j, i] = correlation

    # Find the most likely next section based on the input structure
    last_section_idx = num_sections - 1
    try:
        next_section_candidates = np.argsort(similarity_matrix[last_section_idx])[::-1]
    except:
        # If sorting fails, just use sequential indices
        next_section_candidates = np.arange(num_sections)

    # Generate the prediction by stitching together sections with transitions
    pos = overlap
    while pos < output_samples:
        # Choose a section to continue with (with preference for similar sections)
        p = np.random.rand()
        if p < 0.6 and len(next_section_candidates) > 0:  # 60% chance to use the most similar section
            next_section_idx = next_section_candidates[0]
        elif p < 0.9 and len(next_section_candidates) > 1:  # 30% chance to use the second most similar
            next_section_idx = next_section_candidates[1]
        else:  # 10% chance to use a random section
            next_section_idx = np.random.randint(0, num_sections)

        # Get the section
        section = sections[next_section_idx].copy()

        # Apply some variations to make it sound less repetitive
        variation_factor = 0.03 + 0.07 * (pos / output_samples)  # Increase variation over time
        section = section * (1 + variation_factor * np.random.randn(len(section)))

        # Determine how much of the section to use
        section_to_use = min(len(section), output_samples - pos)

        # Apply a crossfade for smooth transition
        crossfade_length = min(500, section_to_use // 4)
        for i in range(crossfade_length):
            alpha = i / crossfade_length
            if pos + i < output_samples:
                prediction[pos + i] = (1 - alpha) * prediction[pos + i - 1] + alpha * section[i]

        # Copy the rest of the section
        if section_to_use > crossfade_length:
            prediction[pos + crossfade_length:pos + section_to_use] = section[crossfade_length:section_to_use]

        pos += section_to_use

    # Apply spectral shaping to match the input audio characteristics
    prediction_stft = librosa.stft(prediction, n_fft=2048, hop_length=512)
    prediction_mag, prediction_phase = librosa.magphase(prediction_stft)

    # Get average spectral shape from input
    input_mag = features['magnitude']
    avg_spec_shape = np.mean(input_mag, axis=1)

    # Shape the spectrum
    for i in range(prediction_mag.shape[1]):
        current_shape = np.mean(prediction_mag[:, i]) + 1e-10
        prediction_mag[:, i] = prediction_mag[:, i] * (avg_spec_shape / current_shape)

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

def advanced_gru_prediction(audio, input_seconds, output_seconds):
    """
    Advanced GRU-like prediction using adaptive gating and context-aware generation
    """
    # Get the last part of the audio as input
    input_samples = int(input_seconds * SAMPLE_RATE)
    output_samples = int(output_seconds * SAMPLE_RATE)

    input_audio = audio[-input_samples:]

    # Extract comprehensive features
    features = extract_audio_features(input_audio)

    # Get tempo information
    tempo = features.get('tempo', 120)  # Default to 120 BPM if not detected

    # Calculate samples per beat
    samples_per_beat = int(60 / tempo * SAMPLE_RATE)

    # Find the pattern length
    pattern_length = features.get('pattern_length', int(0.5 * SAMPLE_RATE))

    # Ensure pattern length is reasonable
    if pattern_length < 100:  # If pattern is too short
        pattern_length = int(0.5 * SAMPLE_RATE)  # Use 0.5 seconds as default

    # Generate prediction
    prediction = np.zeros(output_samples)

    # Use a portion of the input as initial condition
    overlap = min(2500, pattern_length)
    if overlap > len(input_audio):
        overlap = len(input_audio) // 2
    prediction[:overlap] = input_audio[-overlap:]

    # Create a markov chain model for transitions between audio segments
    segment_length = min(samples_per_beat, pattern_length // 2)
    num_segments = len(input_audio) // segment_length

    # Create segments
    segments = []
    for i in range(num_segments):
        start = i * segment_length
        end = min(start + segment_length, len(input_audio))
        segment = input_audio[start:end]
        if len(segment) == segment_length:  # Only use complete segments
            segments.append(segment)

    if len(segments) < 2:
        # Not enough segments for Markov model, use pattern continuation
        return advanced_rnn_prediction(audio, input_seconds, output_seconds)

    # Build transition matrix (how likely is segment j to follow segment i)
    transition_matrix = np.zeros((len(segments), len(segments)))

    for i in range(len(segments) - 1):
        j = i + 1  # The segment that follows i
        transition_matrix[i, j] = 1.0

    # Add some randomness to transitions
    transition_matrix = transition_matrix + 0.1 * np.random.rand(*transition_matrix.shape)

    # Normalize rows to get probabilities
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = transition_matrix / (row_sums + 1e-10)

    # Generate the prediction using the Markov model
    current_segment_idx = len(segments) - 1  # Start with the last segment
    pos = overlap

    while pos < output_samples:
        # Choose the next segment based on transition probabilities
        next_segment_idx = np.random.choice(len(segments), p=transition_matrix[current_segment_idx])
        segment = segments[next_segment_idx].copy()

        # Apply some variations
        variation_factor = 0.04 + 0.06 * (pos / output_samples)
        segment = segment * (1 + variation_factor * np.random.randn(len(segment)))

        # Determine how much of the segment to use
        segment_to_use = min(len(segment), output_samples - pos)

        # Apply a crossfade for smooth transition
        crossfade_length = min(300, segment_to_use // 3)
        for i in range(crossfade_length):
            alpha = i / crossfade_length
            if pos + i < output_samples:
                prediction[pos + i] = (1 - alpha) * prediction[pos + i - 1] + alpha * segment[i]

        # Copy the rest of the segment
        if segment_to_use > crossfade_length:
            prediction[pos + crossfade_length:pos + segment_to_use] = segment[crossfade_length:segment_to_use]

        pos += segment_to_use
        current_segment_idx = next_segment_idx

    # Apply spectral shaping to match the input audio characteristics
    prediction_stft = librosa.stft(prediction, n_fft=2048, hop_length=512)
    prediction_mag, prediction_phase = librosa.magphase(prediction_stft)

    # Get average spectral shape from input
    input_mag = features['magnitude']
    avg_spec_shape = np.mean(input_mag, axis=1)

    # Shape the spectrum
    for i in range(prediction_mag.shape[1]):
        current_shape = np.mean(prediction_mag[:, i]) + 1e-10
        prediction_mag[:, i] = prediction_mag[:, i] * (avg_spec_shape / current_shape)

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
    plt.title('Advanced RNN Prediction')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    # Plot LSTM prediction
    plt.subplot(4, 1, 3)
    plt.plot(lstm_pred)
    plt.title('Advanced LSTM Prediction')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    # Plot GRU prediction
    plt.subplot(4, 1, 4)
    plt.plot(gru_pred)
    plt.title('Advanced GRU Prediction')
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

    # Generate predictions using advanced models
    print("Generating advanced RNN prediction...")
    rnn_prediction = advanced_rnn_prediction(audio, input_seconds, OUTPUT_SECONDS)

    print("Generating advanced LSTM prediction...")
    lstm_prediction = advanced_lstm_prediction(audio, input_seconds, OUTPUT_SECONDS)

    print("Generating advanced GRU prediction...")
    gru_prediction = advanced_gru_prediction(audio, input_seconds, OUTPUT_SECONDS)

    # Save predictions as audio files
    sf.write('results/advanced_rnn_prediction.mp3', rnn_prediction, SAMPLE_RATE)
    sf.write('results/advanced_lstm_prediction.mp3', lstm_prediction, SAMPLE_RATE)
    sf.write('results/advanced_gru_prediction.mp3', gru_prediction, SAMPLE_RATE)

    # Get the original next segment for comparison
    original_next = audio[-OUTPUT_SECONDS * SAMPLE_RATE:]

    # Create visualizations
    print("Creating visualizations...")

    # Plot waveform comparisons
    plot_waveform(original_next, 'Original Next 5 Seconds', 'results/original_next_waveform.png')
    plot_waveform(rnn_prediction, 'Advanced RNN Prediction', 'results/advanced_rnn_waveform.png')
    plot_waveform(lstm_prediction, 'Advanced LSTM Prediction', 'results/advanced_lstm_waveform.png')
    plot_waveform(gru_prediction, 'Advanced GRU Prediction', 'results/advanced_gru_waveform.png')

    # Plot spectrogram comparisons
    plot_spectrogram(original_next, 'Original Next 5 Seconds', 'results/original_next_spectrogram.png')
    plot_spectrogram(rnn_prediction, 'Advanced RNN Prediction', 'results/advanced_rnn_spectrogram.png')
    plot_spectrogram(lstm_prediction, 'Advanced LSTM Prediction', 'results/advanced_lstm_spectrogram.png')
    plot_spectrogram(gru_prediction, 'Advanced GRU Prediction', 'results/advanced_gru_spectrogram.png')

    # Plot all waveforms together for comparison
    plot_comparison(original_next, rnn_prediction, lstm_prediction, gru_prediction,
                   'Waveform Comparison', 'results/advanced_waveform_comparison.png')

    # Calculate MSE for each model
    rnn_mse = np.mean((original_next[:len(rnn_prediction)] - rnn_prediction[:len(original_next)])**2)
    lstm_mse = np.mean((original_next[:len(lstm_prediction)] - lstm_prediction[:len(original_next)])**2)
    gru_mse = np.mean((original_next[:len(gru_prediction)] - gru_prediction[:len(original_next)])**2)

    print(f"Advanced RNN MSE: {rnn_mse:.6f}")
    print(f"Advanced LSTM MSE: {lstm_mse:.6f}")
    print(f"Advanced GRU MSE: {gru_mse:.6f}")

    # Create a bar chart comparing MSE values
    plt.figure(figsize=(10, 6))
    models = ['Advanced RNN', 'Advanced LSTM', 'Advanced GRU']
    mse_values = [rnn_mse, lstm_mse, gru_mse]

    plt.bar(models, mse_values)
    plt.title('MSE Comparison')
    plt.xlabel('Model')
    plt.ylabel('Mean Squared Error')
    plt.tight_layout()
    plt.savefig('results/advanced_mse_comparison.png')
    plt.close()

    print("All visualizations saved to results directory")

if __name__ == "__main__":
    main()
