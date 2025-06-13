import numpy as np
import collections
import whisper
import torch
import os
from datetime import datetime
from scipy.io.wavfile import write as write_wav # For saving audio files
from typing import Tuple, Optional

class VADAudioProcessor:
    def __init__(self, config: dict, whisper_model):
        self.config = config
        self.whisper_model = whisper_model
        self.audio_buffer = collections.deque()
        self.is_speaking = False
        self.silence_counter = 0

    def is_speech_chunk(self, frame_data: np.ndarray) -> bool:
        rms = np.sqrt(np.mean(frame_data**2))
        print(f"RMS: {rms:.4f}")        # Debugging line to see RMS values
        return rms > self.config["SILENCE_THRESHOLD_RMS"]

    def process_audio_chunk(self, audio_chunk: np.ndarray) -> Optional[Tuple[str, str]]:
        """
        Processes an audio chunk.
        If an utterance is complete, it triggers transcription and returns a tuple
        containing the saved audio file path and the transcribed text.
        Otherwise, it returns None.
        """
        is_speech = self.is_speech_chunk(audio_chunk)
        
        if is_speech:
            if not self.is_speaking:
                self.is_speaking = True
            self.audio_buffer.append(audio_chunk)
            self.silence_counter = 0
            return None # Not done speaking yet
        
        elif self.is_speaking:
            self.audio_buffer.append(audio_chunk)
            self.silence_counter += 1
            
            chunks_per_second = 10 
            padding_chunks_needed = (self.config["VAD_PADDING_MS"] / 1000) * chunks_per_second
            
            if self.silence_counter >= padding_chunks_needed:
                self.is_speaking = False
                return self._transcribe_and_save_buffer() # End of utterance detected
        
        return None # No transcript ready yet

    def _transcribe_and_save_buffer(self) -> Optional[Tuple[str, str]]:
        """
        Private method that handles the core logic:
        1. Saves the buffered audio to a .wav file.
        2. Transcribes the audio using Whisper.
        3. Returns the file path and the transcription text.
        """
        if not self.audio_buffer:
            return None

        full_audio_np = np.concatenate(list(self.audio_buffer))
        self.audio_buffer.clear()

        min_samples = int((self.config["MIN_SPEECH_DURATION_MS"] / 1000.0) * self.config["SAMPLE_RATE"])
        if len(full_audio_np) < min_samples:
            return None # Discard if too short
        
        # --- File Saving and Transcription Logic ---
        try:
            # 1. Create a unique filename for the audio file
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
            filename = f"utterance_{timestamp}.wav"
            file_path = os.path.join(self.config["OUTPUT_DIR"], filename)

            # 2. Save the audio buffer to the .wav file
            # Convert float32 array back to int16 for standard WAV format
            wav_data = (full_audio_np * 32767).astype(np.int16)
            write_wav(file_path, self.config["SAMPLE_RATE"], wav_data)

            # 3. Transcribe the audio
            audio_to_transcribe = full_audio_np.astype(np.float32)
            result = self.whisper_model.transcribe(
                audio_to_transcribe, 
                fp16=torch.cuda.is_available()
            )
            transcribed_text = result.get("text", "").strip()
            
            # 4. Return the file path and the transcribed text as a tuple
            return file_path, transcribed_text
            
        except Exception as e:
            print(f"Error during transcription or file saving: {e}")
            return None