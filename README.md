# Audio Prediction Models

This project implements three different recurrent neural network architectures (RNN, LSTM, and GRU) for audio prediction. The models are trained on audio data to predict the next segment of audio based on previous segments.

## Project Overview

The goal of this project is to compare the performance of different recurrent neural network architectures for audio prediction tasks. The models are trained to predict the next 5 seconds of audio based on the previous audio context.

### Models Implemented

1. **Vanilla RNN (rnn_model.py)**: A basic recurrent neural network implementation for audio prediction.
2. **LSTM (lstm_model.py)**: Long Short-Term Memory network, which is better at capturing long-term dependencies in sequential data.
3. **GRU (gru_model.py)**: Gated Recurrent Unit network, which is a simplified version of LSTM with fewer parameters.

## Dataset

The project uses audio files (MP3 format) as input data. Two sample audio files are included:
- `input.mp3`: Primary audio file for training and testing
- `input2.mp3`: Secondary audio file for additional testing

## Technical Details

### Audio Processing

- **Sample Rate**: 16kHz
- **Feature Extraction**: Mel spectrogram with 128 features
- **Hop Length**: 512 samples
- **Prediction Length**: Fixed at 5 seconds

### Model Architecture

All three models follow a similar architecture:
1. Input audio is converted to mel spectrogram features
2. Features are normalized and segmented
3. The recurrent network processes the input sequence
4. A fully connected layer maps the final hidden state to the output
5. The output is converted back to audio using the Griffin-Lim algorithm

### Training Process

1. The audio file is loaded and preprocessed
2. The entire audio except the last 5 seconds is used as training data
3. The model is trained to predict the next segment of audio
4. Mean Squared Error (MSE) is used as the loss function
5. The Adam optimizer is used for training

## Requirements

- Python 3.8+
- PyTorch
- Librosa
- NumPy
- Matplotlib
- SoundFile

## Usage

### Running the RNN Model

```bash
python rnn_model.py
```

This will:
1. Load the audio file (`input.mp3`)
2. Train the RNN model
3. Generate a prediction for the next 5 seconds
4. Save the prediction and visualizations to the `results` directory

### Running the LSTM Model

```bash
python lstm_model.py
```

This will perform the same steps as above but using the LSTM architecture.

### Running the GRU Model

```bash
python gru_model.py
```

This will perform the same steps as above but using the GRU architecture.

## Results

Each model generates:
1. An audio file with the predicted next 5 seconds
2. Waveform visualizations comparing the original and predicted audio
3. Spectrogram visualizations comparing the original and predicted audio
4. Mean Squared Error (MSE) calculation between the original and predicted audio

## Model Comparison

### Vanilla RNN
- **Strengths**: Simplest architecture, fastest training time
- **Weaknesses**: Limited ability to capture long-term dependencies
- **Best for**: Short audio segments with simple patterns

### LSTM
- **Strengths**: Better at capturing long-term dependencies, handles vanishing gradient problem
- **Weaknesses**: More complex architecture, slower training time
- **Best for**: Complex audio patterns with long-term dependencies

### GRU
- **Strengths**: Simpler than LSTM but still captures long-term dependencies, faster training than LSTM
- **Weaknesses**: May not perform as well as LSTM on very complex patterns
- **Best for**: Balance between performance and training efficiency

## Performance Metrics

The models are evaluated using Mean Squared Error (MSE) between the original and predicted audio. Lower MSE indicates better prediction quality.

## Visualizations

Each model generates two types of visualizations:
1. **Waveform Comparison**: Shows the amplitude of the original and predicted audio over time
2. **Spectrogram Comparison**: Shows the frequency content of the original and predicted audio over time

## Limitations and Future Work

### Current Limitations
- Fixed prediction length (5 seconds)
- Limited to single audio file training
- No real-time prediction capability

### Potential Improvements
- Implement a more sophisticated training process with validation
- Add support for batch processing of multiple audio files
- Explore transformer-based architectures for audio prediction
- Implement real-time audio prediction
- Add support for different audio formats and quality settings

## Conclusion

This project demonstrates the application of different recurrent neural network architectures for audio prediction tasks. The comparison between RNN, LSTM, and GRU models provides insights into their relative strengths and weaknesses for this specific application.

## Author

Deepika Ambade

## License

This project is open source and available under the MIT License.
