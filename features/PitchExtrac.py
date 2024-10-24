import os
import librosa
import numpy as np
import pandas as pd


def extract_features_segment(y_segment, sr):
    # Extract features for the segment
    chromagram = librosa.feature.chroma_stft(y=y_segment, sr=sr).mean(axis=1)
    rms = librosa.feature.rms(y=y_segment).mean(axis=1)
    spectral_centroid = librosa.feature.spectral_centroid(y=y_segment, sr=sr).mean(axis=1)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_segment, sr=sr).mean(axis=1)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y_segment, sr=sr).mean(axis=1)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y_segment).mean(axis=1)
    mfcc = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=20).mean(axis=1)

    # Combine all features into a single dictionary
    features = {
        'Chromagram': chromagram,
        'RMS': rms,
        'Spectral_Centroid': spectral_centroid,
        'Spectral_Bandwidth': spectral_bandwidth,
        'Spectral_Rolloff': spectral_rolloff,
        'Zero_Crossing_Rate': zero_crossing_rate,
    }

    # Add MFCC coefficients
    for i in range(1, 21):
        features[f'MFCC_{i}'] = mfcc[i - 1]

    return features


def process_audio_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.flac'):
                file_path = os.path.join(dirpath, filename)

                # Load the audio file
                y, sr = librosa.load(file_path, sr=None)

                # Remove silence from the beginning and end
                y, _ = librosa.effects.trim(y)

                # Determine the number of samples per second
                samples_per_second = sr

                # Calculate the total duration of the audio in seconds
                total_seconds = len(y) // samples_per_second

                # List to hold feature data
                data = []

                # Process each second of the audio
                for second in range(total_seconds):
                    start_sample = second * samples_per_second
                    end_sample = start_sample + samples_per_second

                    # Extract the 1-second segment
                    y_segment = y[start_sample:end_sample]

                    # Extract features for this segment
                    features = extract_features_segment(y_segment, sr)

                    # Prepare row data including time (in seconds)
                    row_data = [second] + list(features['Chromagram']) + list(features['RMS']) + list(
                        features['Spectral_Centroid']) + list(features['Spectral_Bandwidth']) + list(
                        features['Spectral_Rolloff']) + list(features['Zero_Crossing_Rate']) + [features[f'MFCC_{i}']
                                                                                                for i in range(1, 21)]

                    # Append the row data to the data list
                    data.append(row_data)

                # Define column names
                columns = ['Time'] + [f'Chromagram_{i + 1}' for i in range(12)] + ['RMS'] + ['Spectral_Centroid'] + [
                    'Spectral_Bandwidth'] + ['Spectral_Rolloff'] + ['Zero_Crossing_Rate'] + [f'MFCC_{i}' for i in
                                                                                             range(1, 21)]

                # Convert the list of features into a DataFrame
                df = pd.DataFrame(data, columns=columns)

                # Save CSV with the same name as the audio file
                csv_filename = filename.replace('.flac', '.csv')
                csv_path = os.path.join(dirpath, csv_filename)

                # Save the features to CSV
                df.to_csv(csv_path, index=False)
                print(f"Processed {file_path} -> {csv_path}")


if __name__ == "__main__":
    root_directory = "path_to_your_root_directory"
    process_audio_files(root_directory)

if __name__ == "__main__":
    root_directory = "F:/AudioCorpus/pt-br/TrueCorpus/mls_portuguese/train/audio"
    process_audio_files(root_directory)
