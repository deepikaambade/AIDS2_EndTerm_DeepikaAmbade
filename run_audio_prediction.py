import torch
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import time
import argparse
from audio_prediction_model import (
    preprocess_audio, AudioPredictionDataset, VanillaRNN, LSTMModel, GRUModel,
    train_model, generate_prediction, plot_waveform_comparison, 
    plot_spectrogram_comparison, create_output_dir, SAMPLE_RATE, FEATURE_DIM, HOP_LENGTH
)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Audio Prediction with RNN Models')
    parser.add_argument('--audio_file', type=str, default='input.mp3', help='Input audio file path')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--output_seconds', type=int, default=5, help='Length of prediction in seconds')
    parser.add_argument('--models', type=str, default='all', help='Models to train: rnn, lstm, gru, or all')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = create_output_dir()
    
    # Load audio file (use the entire file)
    print(f"Loading audio file: {args.audio_file}")
    audio = preprocess_audio(args.audio_file)
    audio_duration = len(audio) / SAMPLE_RATE
    print(f"Audio duration: {audio_duration:.2f} seconds")
    
    # Set input seconds to be the entire audio except the last output_seconds
    input_seconds = audio_duration - args.output_seconds
    print(f"Using {input_seconds:.2f} seconds as input context")
    print(f"Predicting {args.output_seconds} seconds of audio")
    
    # Convert seconds to frames
    input_frames = int(input_seconds * SAMPLE_RATE / HOP_LENGTH)
    output_frames = int(args.output_seconds * SAMPLE_RATE / HOP_LENGTH)
    
    # Create dataset and dataloader
    dataset = AudioPredictionDataset(
        audio, 
        input_len=input_frames,
        output_len=output_frames
    )
    
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize models
    input_dim = FEATURE_DIM
    hidden_dim = 256
    output_dim = FEATURE_DIM * output_frames
    
    # Determine which models to train
    models_to_train = []
    if args.models.lower() == 'all' or 'rnn' in args.models.lower():
        models_to_train.append(('RNN', VanillaRNN(input_dim, hidden_dim, output_dim)))
    if args.models.lower() == 'all' or 'lstm' in args.models.lower():
        models_to_train.append(('LSTM', LSTMModel(input_dim, hidden_dim, output_dim)))
    if args.models.lower() == 'all' or 'gru' in args.models.lower():
        models_to_train.append(('GRU', GRUModel(input_dim, hidden_dim, output_dim)))
    
    # Dictionary to store results
    results = {}
    
    # Train models and generate predictions
    for model_name, model in models_to_train:
        print(f"\nTraining {model_name} model...")
        start_time = time.time()
        
        # Train model
        losses = train_model(model, train_loader, model_name, epochs=args.epochs)
        
        # Calculate training time
        training_time = time.time() - start_time
        print(f"{model_name} training completed in {training_time:.2f} seconds")
        
        # Use the entire audio except the last output_seconds as input for prediction
        input_audio = audio[:-args.output_seconds * SAMPLE_RATE]
        
        # Generate prediction
        print(f"Generating {model_name} prediction...")
        prediction = generate_prediction(model, input_audio, input_frames, output_frames)
        
        # Save prediction as audio file
        prediction_file = os.path.join(output_dir, f'{model_name.lower()}_prediction.mp3')
        sf.write(prediction_file, prediction, SAMPLE_RATE)
        print(f"Prediction saved to {prediction_file}")
        
        # Get the original next segment for comparison
        original_next = audio[-args.output_seconds * SAMPLE_RATE:]
        
        # Calculate MSE
        min_len = min(len(original_next), len(prediction))
        mse = np.mean((original_next[:min_len] - prediction[:min_len])**2)
        print(f"{model_name} MSE: {mse:.6f}")
        
        # Create visualizations
        print(f"Creating {model_name} visualizations...")
        plot_waveform_comparison(
            original_next, 
            prediction, 
            model_name
        )
        
        # Try to import librosa.display for spectrogram visualization
        try:
            import librosa.display
            plot_spectrogram_comparison(
                original_next, 
                prediction, 
                model_name
            )
        except ImportError:
            print("librosa.display not available, skipping spectrogram visualization")
        
        # Store results
        results[model_name] = {
            'mse': mse,
            'training_time': training_time,
            'final_loss': losses[-1]
        }
    
    # Compare model performance
    if len(results) > 1:
        print("\nModel Performance Comparison:")
        print("-" * 60)
        print(f"{'Model':<10} {'MSE':<15} {'Training Time (s)':<20} {'Final Loss':<15}")
        print("-" * 60)
        
        for model_name, metrics in results.items():
            print(f"{model_name:<10} {metrics['mse']:<15.6f} {metrics['training_time']:<20.2f} {metrics['final_loss']:<15.6f}")
        
        # Create comparison bar chart
        plt.figure(figsize=(12, 6))
        
        # MSE comparison
        plt.subplot(1, 2, 1)
        plt.bar(results.keys(), [metrics['mse'] for metrics in results.values()])
        plt.title('MSE Comparison')
        plt.ylabel('Mean Squared Error')
        
        # Training time comparison
        plt.subplot(1, 2, 2)
        plt.bar(results.keys(), [metrics['training_time'] for metrics in results.values()])
        plt.title('Training Time Comparison')
        plt.ylabel('Time (seconds)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
        plt.close()

if __name__ == "__main__":
    main()
