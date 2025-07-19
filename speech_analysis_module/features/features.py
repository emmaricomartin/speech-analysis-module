import parselmouth
import librosa
import numpy as np

def extract_features(audio, sr, praat_voice_report=False):
    """
    Extract pitch, jitter, shimmer, and harmonicity using Parselmouth (Praat)
    and additional features using librosa.

    Parameters:
    - audio: np.ndarray, audio signal
    - sr: int, sample rate
    - praat_voice_report: bool, whether to return full Praat voice report

    Returns:
    - dict: extracted features
    """
    features = {}

    try:
        snd = parselmouth.Sound(audio, sampling_frequency=sr)
    except Exception as e:
        print(f"[Warning] Could not convert to Parselmouth Sound: {e}")
        snd = None

    # ---- Parselmouth features ----

    # Pitch
    try:
        if snd:
            pitch_obj = snd.to_pitch()
            pitch_values = pitch_obj.selected_array['frequency']
            pitch_values = pitch_values[pitch_values > 0]  # remove unvoiced
            features['pitch_mean'] = np.mean(pitch_values) if len(pitch_values) > 0 else np.nan
            features['pitch_std'] = np.std(pitch_values) if len(pitch_values) > 0 else np.nan
        else:
            raise ValueError("snd is None")
    except Exception as e:
        print(f"[Warning] Pitch extraction failed: {e}")
        features['pitch_mean'] = np.nan
        features['pitch_std'] = np.nan

    # Harmonicity
    try:
        if snd:
            harm = snd.to_harmonicity_cc()
            values = harm.values[harm.values != -200]
            features['harmonicity_mean'] = np.mean(values) if len(values) > 0 else np.nan
        else:
            raise ValueError("snd is None")
    except Exception as e:
        print(f"[Warning] Harmonicity extraction failed: {e}")
        features['harmonicity_mean'] = np.nan

    # Jitter & Shimmer
    try:
        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
        num_points = parselmouth.praat.call(point_process, "Get number of points")
        if num_points == 0:
            raise ValueError("No periodic points detected")
    
        features['jitter_local'] = parselmouth.praat.call([snd, point_process], "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        features['shimmer_local'] = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    except Exception as e:
        print(f"[Warning] Jitter/Shimmer extraction failed: {e}")
        features['jitter_local'] = np.nan
        features['shimmer_local'] = np.nan

    # Praat voice report
    if praat_voice_report:
        try:
            if snd and 'pitch_obj' in locals():
                report = parselmouth.praat.call(snd, pitch_obj, "Voice report", 0, 0, 75, 500, 1.3, 1.6, 0.03, 0.45)
                features['praat_voice_report'] = report
            else:
                raise ValueError("snd or pitch_obj not available")
        except Exception as e:
            print(f"[Warning] Voice report extraction failed: {e}")
            features['praat_voice_report'] = None

    # ---- Librosa features ----

    # Zero Crossing Rate
    try:
        features['zcr_mean'] = np.mean(librosa.feature.zero_crossing_rate(audio)[0])
    except Exception as e:
        print(f"[Warning] ZCR extraction failed: {e}")
        features['zcr_mean'] = np.nan

    # Spectral Centroid
    try:
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
    except Exception as e:
        print(f"[Warning] Spectral centroid extraction failed: {e}")
        features['spectral_centroid_mean'] = np.nan

    # Spectral Bandwidth
    try:
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    except Exception as e:
        print(f"[Warning] Spectral bandwidth extraction failed: {e}")
        features['spectral_bandwidth_mean'] = np.nan

    # Spectral Rolloff
    try:
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    except Exception as e:
        print(f"[Warning] Spectral rolloff extraction failed: {e}")
        features['spectral_rolloff_mean'] = np.nan

    # RMS Energy
    try:
        rms = librosa.feature.rms(y=audio)[0]
        features['rms_mean'] = np.mean(rms)
    except Exception as e:
        print(f"[Warning] RMS extraction failed: {e}")
        features['rms_mean'] = np.nan

    # MFCCs (first 3)
    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        for i in range(3):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
    except Exception as e:
        print(f"[Warning] MFCC extraction failed: {e}")
        for i in range(3):
            features[f'mfcc_{i+1}_mean'] = np.nan

    return features

