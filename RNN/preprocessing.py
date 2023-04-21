import collections
import contextlib
import wave
import librosa
import os
import numpy as np
from scipy.io import wavfile
from scipy import signal
import webrtcvad
from pydub import AudioSegment
import shutil


# This part is originated from https://github.com/wiseman/py-webrtcvad/blob/master/example.py

def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
   
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])


#------------------------------------------------------------------------------------------------------------------------------

# These functions are used for preprocessing audio files


def perform_vad(wav_file_path, new_file_path):
    
    """
    Performs voice activity detection on a wav file.
    Parameters:
    wav_file_path: path to the wav file
    new_file_path: path to the new wav file
    """
   
    audio, sample_rate = read_wave(wav_file_path)
    vad = webrtcvad.Vad(1)
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 30, 300, vad, frames)
    
    k = 0    

    for i, segment in enumerate(segments):
        write_wave(new_file_path, segment, sample_rate)
        k += 1
    
    if k == 0:
        shutil.copy(wav_file_path, new_file_path)

    return k


def resample_wav(wav_file_path, new_file_path, new_sample_rate):
    
    """
    Resamples a wav file to a new sample rate.
    Parameters:
    wav_file_path: path to the wav file
    new_file_path: path to the new wav file
    new_sample_rate: new sample rate
    """

    sample_rate, samples = wavfile.read(wav_file_path)
    resampled = signal.resample(samples, int(new_sample_rate/sample_rate * samples.shape[0]))
    wavfile.write(new_file_path, new_sample_rate, resampled.astype(np.int16))


def padding(wav_file_path, new_path, pad_ms=1000):
    
    """
    Pads a wav file with silence.
    Parameters:
    wav_file_path: path to the wav file
    new_file_path: path to the new wav file
    pad_ms: clip duration in milliseconds after padding
    """

    audio = AudioSegment.from_wav(wav_file_path)
    silence = AudioSegment.silent(duration=pad_ms-len(audio)+1)
    

    padded = audio + silence
    padded.export(new_path, format='wav')

def cut_wav_into_clips(wav_file_path, new_file_path, name, clip_duration_ms=1000):
    
    """
    Cuts a wav file to clips of a given duration.
    Parameters:
    wav_file_path: path to the wav file
    new_file_path: path to the new wav file
    name: name of the new clips
    clip_duration_ms: clip duration in milliseconds
    """

    audio = AudioSegment.from_wav(wav_file_path)
    for i, chunk in enumerate(audio[::clip_duration_ms]):
        chunk.export(f"{new_file_path}\\{name + str(i)}" + '.wav', format='wav')

def extract_features(wav_file_path, mfccs=True, deltas=True, delta_deltas=True):
    """
    Extracts features from a wav file.
    Parameters:
    wav_file_path: path to the wav file
    Returns:
    features: features
    label: label of the wav file
    """

    audio, sample_rate = librosa.load(wav_file_path)

    if mfccs:
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    if deltas:
        delta = librosa.feature.delta(mfccs)
    if delta_deltas:
        delta_delta = librosa.feature.delta(mfccs, order=2)
    features = np.concatenate([mfccs, delta, delta_delta], axis=0)
   
    return features

if __name__ == '__main__':
    wav_file = 'samples\\sample.wav'
    wav_file2 = 'working_sample.wav'
    perform_vad(wav_file, wav_file2)
    padding(wav_file2, wav_file2, 1000)
    resample_wav(wav_file2, wav_file2, 8000)
