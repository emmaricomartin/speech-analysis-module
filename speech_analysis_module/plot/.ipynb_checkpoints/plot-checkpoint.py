import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

#plot oscilograma
def oscilo(raw, sr):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(raw, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(False) #show grid
    plt.show()

#plot spectrogram
def spectro(stft, sr):
    plt.figure(figsize=(10, 6))
    # Convert the magnitude spectrogram to dB scale using the corrected approach
    magnitude_spectrogram = np.abs(stft)  # Get the magnitude of the STFT
    db_spectrogram = librosa.power_to_db(magnitude_spectrogram**2, ref=np.max)  # Convert to dB scale
    # Plot the spectrogram
    librosa.display.specshow(db_spectrogram, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (dB scale)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.grid(False)  # Show grid
    plt.show()

#plot psd
def density(psd_values, freqs, db=False):
    plt.figure(figsize=(10, 6))
    if db:
        plt.plot(freqs, psd_values)  # Use linear plot since values are already in dB
        plt.ylabel('Power/Frequency (dB/Hz)')
    else:
        plt.semilogy(freqs, psd_values)  # Use semilog for linear power values
        plt.ylabel('Power/Frequency (linear scale)')
    
    plt.xlabel('Frequency (Hz)')
    plt.title('Power Spectral Density (PSD) using Welch\'s Method')
    plt.grid()
    plt.show()