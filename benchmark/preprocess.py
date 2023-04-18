import librosa
import numpy as np

def audio_preprocess(audio_file: str):
	audio, sr = librosa.load(audio_file, sr=None)
	new_sr = 16000
	audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=new_sr)
	return np.asarray(audio_resampled)