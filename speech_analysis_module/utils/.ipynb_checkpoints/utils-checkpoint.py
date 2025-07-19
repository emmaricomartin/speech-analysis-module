import os
import numpy as np
import soundfile as sf
import whisper
import sounddevice as sd
import tempfile

#Get the list of files of a folder
def get_files(directory, extensions=[".wav"], show=True):
    if extensions:
        file_list = [f for f in os.listdir(directory) if any(f.endswith(ext) for ext in extensions)]
    else:
        file_list = os.listdir(directory)

    if show:
        print(f"Found {len(file_list)} files: {file_list}")
    return file_list

#load audio
def load_audio(filepath):
    audio, samplerate = sf.read(filepath)
    return audio, samplerate

#Resample
def resample(audio, sr, new_sr, verbose=True):
    if verbose:
        print(f"Resampling from {sr} Hz to {new_sr} Hz...")
    
    resampled_audio = librosa.resample(y=audio, orig_sr=sr, target_sr=new_sr)
    
    return resampled_audio, new_sr

#Play the audio file on your output device
def play(audio, sr):
    print(f"Playing audio... Duration: {round(len(audio) / sr, 2)} seconds")
    sd.play(audio, sr)
    sd.wait()  # Wait until playback is finished

# Save function
def save_audio(audio, sr, filename, output_directory):

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created directory: {output_directory}")

    output_path = os.path.join(output_directory, filename)
    sf.write(output_path, audio, sr)
    print(f"Saved processed audio as: {output_path}")
    
    return output_path

#Transcribe with whisper
def to_text(file_path, model="base", device="cpu", language=None, verbose=True):

    audio_data, sample_rate = sf.read(file_path)
    audio_data = audio_data.astype(np.float32)
    model = whisper.load_model(model, device = device)
    transcription_result = model.transcribe(audio_data, language=language)
    transcription_text = transcription_result["text"]
    if verbose==True:
        print(transcription_text)
    else:
        pass
    
    return(transcription_text)

#Transcribe an array to text with whisper
def array_to_text(input_audio, sr=None, model="base", language=None, verbose=True):
    
    # Load model once
    model = whisper.load_model(model)
    
    # If input is a NumPy array
    if isinstance(input_audio, np.ndarray):
        if sr is None:
            raise ValueError("Sample rate (sr) must be provided when using raw audio array.")
        # Save to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, input_audio, sr)
            audio_path = tmp_file.name
    elif isinstance(input_audio, str):
        audio_path = input_audio
    else:
        raise TypeError("input_audio must be a file path or a NumPy array.")

    # Transcribe
    result = model.transcribe(audio_path, language=language)
    text = result['text']

    # Clean up temporary file if it was created
    if isinstance(input_audio, np.ndarray):
        os.remove(audio_path)
    
    if verbose:
        print(text)
        
    return text
    

