import uvicorn
import numpy as np
import torch
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict

# Import our VAD logic class from the separate processor file
from .processor import VADAudioProcessor 

# Centralized configuration. Note the absence of CHUNK_DURATION_MS as it's
# now determined by the frontend's audio buffer size.
CONFIG = {
    "SAMPLE_RATE": 16000,
    "WHISPER_MODEL_NAME": "base.en",
    "VAD_PADDING_MS": 700,
    "MIN_SPEECH_DURATION_MS": 300,
    #"SILENCE_THRESHOLD_RMS": 0.04,
     "OUTPUT_DIR": "transcribed_audio"
     # This is the most important parameter to tune!
}

# --- FastAPI App Initialization ---
app = FastAPI()
app.add_middleware(    #making sure it is accessible from any origin
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# --- App State and Events ---
@app.on_event("startup")  # on startup performing the model loading 
def load_whisper_model():
    """Load the Whisper model into memory at application startup."""
    import whisper
    print("Loading Whisper model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Whisper will use device: {device}")
    app.state.whisper_model = whisper.load_model(CONFIG["WHISPER_MODEL_NAME"], device=device)
    print("Whisper model loaded successfully.")

    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    print(f"Audio and text outputs will be saved to '{CONFIG['OUTPUT_DIR']}/'")

# This dictionary holds a processor instance for each connected client.
clients: Dict[WebSocket, VADAudioProcessor] = {}

# --- WebSocket Endpoint ---
@app.websocket("/listen")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections for real-time transcription."""
    await websocket.accept()
    processor = VADAudioProcessor(config=CONFIG, whisper_model=app.state.whisper_model) #creates an object that can be used by multiple models
    clients[websocket] = processor
    print(f"Client connected. Total clients: {len(clients)}")

    try:
        while True:
            # Receive raw 16-bit PCM audio bytes from the client
            pcm_bytes = await websocket.receive_bytes()

            # Convert the raw bytes directly into a NumPy array.
            # The audio is already in the correct format, so no FFmpeg is needed.
            audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Pass the audio chunk to the client's dedicated VAD processor
            transcribed_text = processor.process_audio_chunk(audio_np)
            
            # If the processor returns a final transcript, send it back to the client
            if transcribed_text:
                print(f"Transcription: {transcribed_text}")
                await websocket.send_text(transcribed_text)

    except WebSocketDisconnect:
        print(f"Client disconnected.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Clean up the client's state when they disconnect
        if websocket in clients:
            del clients[websocket]
            print(f"Client state cleaned up. Total clients: {len(clients)}")