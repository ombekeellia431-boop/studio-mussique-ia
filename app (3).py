import os
import time
import uuid
from pathlib import Path
from typing import List, Optional
import uvicorn
import subprocess # Ensure subprocess is imported for fuser command

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

# --- AI Music Generation Dependencies (MusicGen via Hugging Face Transformers) ---
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
import torch # Required by transformers and MusicGen

# --- MoviePy Dependencies ---
# These imports are here because they are used directly in functions defined below.
# MoviePy can sometimes cause issues if imported globally and then reloaded in a notebook
# hence they are imported inside functions for some cases or conditionally where possible.
# For a standalone app.py, global import is fine.
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, ColorClip

# --- MoviePy Functions ---
# Note: TextClip functionality has been removed from create_dummy_video
# to avoid ImageMagick policy issues often encountered in environments like Colab.
# For full functionality, ensure ImageMagick is configured correctly (policy.xml)
# if you need features like TextClip, or ensure the environment is permissive.

def create_dummy_video(output_path="dummy_video.mp4", duration=5, fps=24, size=(640, 480)) -> Optional[str]:
    """
    Creates a dummy MP4 video file for testing purposes (without TextClip).
    """
    print(f"Creating dummy video: {output_path} (without text overlay)...")
    try:
        clip = ColorClip(size, color=(0, 0, 0), duration=duration)
        clip.write_videofile(str(output_path), fps=fps, logger=None)
        print(f"Dummy video created successfully at {output_path}")
        return str(output_path)
    except Exception as e:
        print(f"Error creating dummy video: {e}")
        return None

def create_dummy_audio(output_path="dummy_audio.wav", duration=5, sr=44100, amplitude=0.5) -> Optional[str]:
    """
    Creates a dummy WAV audio file for testing purposes.
    (MP3 conversion often requires FFMPEG system-level setup which can be complex in generic environments).
    """
    import numpy as np
    from scipy.io.wavfile import write as write_wav

    print(f"Creating dummy audio: {output_path}")
    try:
        t = np.linspace(0, duration, int(sr * duration), False)
        data = amplitude * np.sin(2 * np.pi * 440 * t) # 440 Hz sine wave
        scaled_data = np.int16(data * 32767)

        write_wav(str(output_path), sr, scaled_data)
        print(f"Dummy WAV audio created successfully at {output_path}")
        return str(output_path)
    except Exception as e:
        print(f"An error occurred creating dummy audio: {e}")
        return None

def cut_video_clip(input_video_path: str, start_time: float, end_time: float, output_path: str) -> Optional[str]:
    """
    Cuts a video clip from a source file given a start and end time.
    """
    print(f"Cutting video from {input_video_path} from {start_time}s to {end_time}s...")
    try:
        video_clip = VideoFileClip(input_video_path)
        cut_clip = video_clip.subclip(start_time, end_time)
        cut_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
        video_clip.close()
        print(f"Video cut successfully to: {output_path}")
        return str(output_path)
    except Exception as e:
        print(f"Error cutting video: {e}")
        return None

def add_audio_to_video(input_video_path: str, input_audio_path: str, output_path: str) -> Optional[str]:
    """
    Adds an audio track to a video clip.
    """
    print(f"Adding audio from {input_audio_path} to video {input_video_path}...")
    try:
        video_clip = VideoFileClip(input_video_path)
        audio_clip = AudioFileClip(input_audio_path)

        final_clip = video_clip.set_audio(audio_clip)

        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)

        video_clip.close()
        audio_clip.close()
        print(f"Audio added to video successfully. Output: {output_path}")
        return str(output_path)
    except Exception as e:
        print(f"Error adding audio to video: {e}")
        return None

# --- Global Model Loading for MusicGen (Load once at startup) ---
# This ensures the model is loaded only once when the FastAPI app starts,
# improving performance for subsequent generation requests.
print("Loading MusicGen model and processor...")
try:
    musicgen_processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    musicgen_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    print("MusicGen model and processor loaded.")
except Exception as e:
    print(f"Error loading MusicGen model: {e}")
    print("MusicGen features will not be available.")
    musicgen_processor = None
    musicgen_model = None

# --- Project and Job Status Databases (In-memory mocks) ---
PROJECT_DB = {}
JOB_STATUS_DB = {}

def _initialize_mock_project(project_id: str):
    """Initializes a mock project entry if it doesn't exist."""
    if project_id not in PROJECT_DB:
        PROJECT_DB[project_id] = {
            "id": project_id,
            "name": f"Project {project_id}",
            "vocals": [],
            "instrumentals": [],
            "metadata": {"last_updated": ""}
        }
        print(f"Mock project '{project_id}' initialized.")

def add_music_to_project(
    music_track_url: str,
    project_id: str,
    generation_parameters: Optional[dict] = None
) -> dict:
    """
    Mocks the backend function to add a user-selected AI-generated music track
    into a specific project.
    """
    _initialize_mock_project(project_id)

    print(f"Attempting to add music track '{music_track_url}' to project '{project_id}'...")

    instrumental_entry = {
        "id": str(uuid.uuid4()),
        "type": "instrumental",
        "source": "AI-generated",
        "url": music_track_url,
        "added_at": time.ctime(),
        "generation_params": generation_parameters if generation_parameters else {},
        "sync_metadata": {
            "start_offset_ms": 0,
            "tempo_bpm": generation_parameters.get('tempo') if generation_parameters else None,
            "key": generation_parameters.get('key') if generation_parameters else None,
            "volume": 1.0
        }
    }

    PROJECT_DB[project_id]["instrumentals"].append(instrumental_entry)
    PROJECT_DB[project_id]["metadata"]["last_updated"] = time.ctime()

    print(f"Successfully added music track to project '{project_id}'.")

    return {"status": "success", "message": "Music track added to project successfully.", "project_info": PROJECT_DB[project_id]}


# --- Real AI Music Generation Logic (using MusicGen) ---
def generate_music_core(
    genre: str = "pop",
    mood: str = "upbeat",
    tempo: int = 120,
    instrumentation: Optional[List[str]] = None,
    duration: int = 60, # Note: MusicGen max_new_tokens approximates duration
    key: str = "C major",
    style_reference: Optional[str] = None
) -> dict:
    """
    Uses MusicGen to generate instrumental music based on user-defined parameters.
    """
    if instrumentation is None:
        instrumentation = ["piano", "strings"]

    generation_parameters_dict = {
        "genre": genre,
        "mood": mood,
        "tempo": tempo,
        "instrumentation": instrumentation,
        "duration": duration,
        "key": key,
        "style_reference": style_reference
    }

    print(f"Initiating real AI music generation with parameters:")
    print(f"  Genre: {genre}, Mood: {mood}, Tempo: {tempo} BPM")
    print(f"  Instrumentation: {', '.join(instrumentation)}, Duration: {duration}s, Key: {key}")
    if style_reference:
        print(f"  Style Reference: {style_reference}")

    if musicgen_model is None or musicgen_processor is None:
        return {
            "status": "failed",
            "message": "MusicGen model not loaded. Check server logs for errors during startup.",
            "output_url": None,
            "parameters": generation_parameters_dict
        }

    # Map parameters to a text prompt for MusicGen
    text_prompt = f"{mood} {genre} music with {', '.join(instrumentation)}, tempo {tempo} BPM, in {key}."
    if style_reference:
        text_prompt += f" Inspired by {style_reference}."

    print(f"MusicGen text prompt: '{text_prompt}'")

    try:
        inputs = musicgen_processor(
            text=[text_prompt],
            padding=True,
            return_tensors="pt",
        )

        # MusicGen generates tokens which are then decoded to audio.
        # 256 tokens typically correspond to about 5-6 seconds of audio.
        # For precise duration, post-processing (trimming/looping) would be needed,
        # but for this example, we let MusicGen generate a fixed length output.
        audio_values = musicgen_model.generate(**inputs, max_new_tokens=256)
        sampling_rate = musicgen_model.config.audio_encoder.sampling_rate

        output_dir = Path("./generated_music")
        output_dir.mkdir(exist_ok=True)

        unique_filename = f"musicgen_track_{uuid.uuid4().hex}.wav"
        output_path = output_dir / unique_filename

        # Save the audio values as a WAV file
        # MusicGen outputs float32, scipy.io.wavfile.write expects int16 for typical WAV
        # Scale and convert to int16
        audio_data_int16 = (audio_values[0, 0].cpu().numpy() * 32767).astype(np.int16)
        scipy.io.wavfile.write(output_path, rate=sampling_rate, data=audio_data_int16)

        print(f"Real AI music generated and saved to: {output_path}")

        return {
            "status": "completed",
            "output_url": str(output_path),
            "parameters": generation_parameters_dict
        }
    except Exception as e:
        print(f"Error during MusicGen generation: {e}")
        return {
            "status": "failed",
            "message": f"Music generation failed: {e}",
            "output_url": None,
            "parameters": generation_parameters_dict
        }

# --- Mock Vocal Processing Logic ---
def process_vocals_core(
    input_audio_path: str,
    output_dir: str = "./processed_vocals_mocks",
    pitch_correction_level: float = 0.0,
    noise_reduction_strength: float = 0.0,
    harmonize_enabled: bool = False,
    vocal_effect_preset: Optional[str] = None
) -> dict:
    """
    Mocks the core logic for AI vocal processing.

    Args:
        input_audio_path (str): The path to the recorded vocal audio file.
        output_dir (str): The directory where the processed audio file will be saved.
        pitch_correction_level (float): Intensity of pitch correction (0.0 to 1.0).
        noise_reduction_strength (float): Strength of noise reduction (0.0 to 1.0).
        harmonize_enabled (bool): Whether vocal harmonization is enabled.
        vocal_effect_preset (Optional[str]): A vocal effect preset.

    Returns:
        dict: A dictionary containing the status, output URL, and parameters of the processing.
    """
    print(f"Initiating vocal processing for '{input_audio_path}' with parameters:")
    print(f"  Pitch Correction: {pitch_correction_level}")
    print(f"  Noise Reduction: {noise_reduction_strength}")
    print(f"  Harmonization: {harmonize_enabled}")
    print(f"  Vocal Effect Preset: {vocal_effect_preset}")

    # Simulate AI model processing time
    time.sleep(2)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    unique_filename = f"processed_vocal_{uuid.uuid4().hex}.wav"
    output_path = Path(output_dir) / unique_filename

    # Create a dummy file to simulate processed output
    with open(output_path, "w") as f:
        f.write(f"# Mock Processed Vocal Data\n")
        f.write(f"Original: {input_audio_path}\n")
        f.write(f"Pitch Correction: {pitch_correction_level}\n")
        f.write(f"Noise Reduction: {noise_reduction_strength}\n")
        f.write(f"Harmonization: {harmonize_enabled}\n")
        f.write(f"Vocal Effect: {vocal_effect_preset}\n")
        f.write(f"Processed on: {time.ctime()}\n")

    print(f"Mock processed vocal saved to: {output_path}")

    return {
        "status": "completed",
        "output_url": str(output_path),
        "parameters": {
            "input_audio_path": input_audio_path,
            "pitch_correction_level": pitch_correction_level,
            "noise_reduction_strength": noise_reduction_strength,
            "harmonize_enabled": harmonize_enabled,
            "vocal_effect_preset": vocal_effect_preset
        }
    }


# --- FastAPI Application Instance ---

app = FastAPI(
    title="AI Music Creation API",
    description="API for AI-powered music generation, vocal processing, video creation, and sharing.",
    version="1.0.0",
)

# --- Pydantic Models for API Request/Response Bodies ---
class MusicGenerateRequest(BaseModel):
    genre: str = "Pop"
    mood: str = "Energetic"
    tempo: int = 120
    instrumentation: Optional[List[str]] = ["Piano", "Drums"]
    duration: int = 60 # seconds (used for prompt, not exact MusicGen output length)
    key: str = "C Major"
    style_reference: Optional[str] = None

class MusicGenerateResponse(BaseModel):
    job_id: str
    status: str
    message: str

class MusicStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: Optional[int] = None
    output_format: Optional[str] = None
    output_url: Optional[str] = None
    message: Optional[str] = None

class MusicParametersResponse(BaseModel):
    genres: List[str]
    moods: List[str]
    instrumentation_options: List[str]

class VocalProcessRequest(BaseModel):
    input_audio_url: str # URL or path to the input vocal track
    pitch_correction_level: float = 0.0
    noise_reduction_strength: float = 0.0
    harmonize_enabled: bool = False
    vocal_effect_preset: Optional[str] = None

class VocalProcessResponse(BaseModel):
    job_id: str
    status: str
    message: str


# --- API Endpoints ---

@app.post("/api/music/generate", response_model=MusicGenerateResponse, status_code=status.HTTP_202_ACCEPTED)
async def initiate_music_generation(request: MusicGenerateRequest):
    job_id = str(uuid.uuid4())
    JOB_STATUS_DB[job_id] = {
        "status": "pending",
        "progress": 0,
        "message": "Music generation request received.",
        "parameters": request.model_dump()
    }

    print(f"Job {job_id}: Music generation initiated for parameters: {request.model_dump()}")

    from threading import Thread # Import here to ensure it's within the function scope
    def run_generation_in_background():
        try:
            job_parameters = JOB_STATUS_DB[job_id]["parameters"]
            generation_result = generate_music_core(
                genre=job_parameters["genre"],
                mood=job_parameters["mood"],
                tempo=job_parameters["tempo"],
                instrumentation=job_parameters["instrumentation"],
                duration=job_parameters["duration"],
                key=job_parameters["key"],
                style_reference=job_parameters["style_reference"]
            )

            if generation_result["status"] == "completed":
                JOB_STATUS_DB[job_id].update({
                    "status": "completed",
                    "progress": 100,
                    "output_format": "WAV",
                    "output_url": generation_result["output_url"],
                    "message": "Music generation completed successfully."
                })
                # Simulate adding to a project after generation
                project_id = "default_user_project"
                add_music_to_project(generation_result["output_url"], project_id, generation_result["parameters"])
            else:
                JOB_STATUS_DB[job_id].update({
                    "status": "failed",
                    "message": generation_result.get("message", "Music generation failed.")
                })
        except Exception as e:
            print(f"Background generation for job {job_id} failed: {e}")
            JOB_STATUS_DB[job_id].update({
                "status": "failed",
                "message": f"Internal server error during generation: {e}"
            })

    Thread(target=run_generation_in_background).start()

    return MusicGenerateResponse(
        job_id=job_id,
        status="pending",
        message="Music generation initiated. Check status endpoint for updates."
    )

@app.get("/api/music/generate/status/{job_id}", response_model=MusicStatusResponse)
async def get_music_generation_status(job_id: str):
    if job_id not in JOB_STATUS_DB:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job ID not found.")

    status_info = JOB_STATUS_DB[job_id]
    return MusicStatusResponse(job_id=job_id, **status_info)

@app.get("/api/music/parameters", response_model=MusicParametersResponse)
async def get_music_parameters():
    return MusicParametersResponse(
        genres=["Pop", "Rock", "Jazz", "Electronic", "Classical", "Ambient"],
        moods=["Happy", "Sad", "Energetic", "Relaxing", "Mysterious", "Epic"],
        instrumentation_options=["Piano", "Drums", "Guitar", "Bass", "Synth", "Strings", "Saxophone", "Flute"]
    )

@app.post("/api/vocals/process", response_model=VocalProcessResponse, status_code=status.HTTP_202_ACCEPTED)
async def initiate_vocal_processing(request: VocalProcessRequest):
    job_id = str(uuid.uuid4())
    JOB_STATUS_DB[job_id] = {
        "status": "pending",
        "progress": 0,
        "message": "Vocal processing request received.",
        "parameters": request.model_dump()
    }

    print(f"Job {job_id}: Vocal processing initiated for parameters: {request.model_dump()}")

    from threading import Thread
    def run_vocal_processing_in_background():
        try:
            job_parameters = JOB_STATUS_DB[job_id]["parameters"]
            output_dir = "./processed_vocals_mocks"
            processed_result = process_vocals_core(
                input_audio_path=job_parameters["input_audio_url"],
                output_dir=output_dir,
                pitch_correction_level=job_parameters["pitch_correction_level"],
                noise_reduction_strength=job_parameters["noise_reduction_strength"],
                harmonize_enabled=job_parameters["harmonize_enabled"],
                vocal_effect_preset=job_parameters["vocal_effect_preset"]
            )

            if processed_result["status"] == "completed":
                JOB_STATUS_DB[job_id].update({
                    "status": "completed",
                    "progress": 100,
                    "output_format": "WAV",
                    "output_url": processed_result["output_url"],
                    "message": "Vocal processing completed successfully."
                })
            else:
                JOB_STATUS_DB[job_id].update({
                    "status": "failed",
                    "message": "Vocal processing failed."
                })
        except Exception as e:
            print(f"Background vocal processing for job {job_id} failed: {e}")
            JOB_STATUS_DB[job_id].update({
                "status": "failed",
                "message": f"Internal server error during vocal processing: {e}"
            })

    Thread(target=run_vocal_processing_in_background).start()

    return VocalProcessResponse(
        job_id=job_id,
        status="pending",
        message="Vocal processing initiated. Check status endpoint for updates."
    )

@app.get("/api/vocals/process/status/{job_id}", response_model=MusicStatusResponse)
async def get_vocal_processing_status(job_id: str):
    if job_id not in JOB_STATUS_DB:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job ID not found.")

    status_info = JOB_STATUS_DB[job_id]
    return MusicStatusResponse(job_id=job_id, **status_info)


@app.get("/app_status")
async def app_status():
    return {"status": "running", "message": "AI Music Creation API is operational.", "job_db_size": len(JOB_STATUS_DB), "project_db_size": len(PROJECT_DB)}


# --- Asynchronous Server Startup and Testing Client Code (for Colab environment) ---

# This section ensures the FastAPI server is started in a background thread
# and then initiates a test request to validate the MusicGen integration.

# --- Force Kill Existing Process on Port 8000 (for Colab restarts) ---
print("\nAttempting to terminate any existing process on port 8000...")
try:
    # Find PIDs using port 8000
    pids_output = subprocess.check_output(["fuser", "8000/tcp"]).decode().strip()
    pids = [pid for pid in pids_output.split() if pid.isdigit()]

    if pids:
        print(f"Found processes on port 8000 with PIDs: {pids}. Terminating...")
        for pid in pids:
            subprocess.run(["kill", "-9", pid])
        print("Processes terminated.")
        time.sleep(2) # Give a moment for port to be released
    else:
        print("No processes found on port 8000.")
except subprocess.CalledProcessError:
    print("No processes found on port 8000, or 'fuser' command not found. Proceeding.")
except Exception as e:
    print(f"Error during process termination: {e}")


# --- Start FastAPI app in a separate thread ---
# nest_asyncio.apply() allows uvicorn's event loop to run in a Colab notebook
nest_asyncio.apply()

def run_fastapi_app_in_thread():
    """Runs the FastAPI app using uvicorn."""
    # The 'app' object is defined in this same cell, so it's directly accessible
    # log_level="warning" to reduce verbose output in Colab
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

print("Launching FastAPI application in a background thread...")
api_thread = threading.Thread(target=run_fastapi_app_in_thread)
api_thread.daemon = True # Allows main program to exit even if thread is running
api_thread.start()
# Give the server a moment to start and load models. MusicGen can take time.
print("FastAPI application launched! Waiting for startup...")
time.sleep(20) # Increased sleep to allow full MusicGen model loading
print("FastAPI application should now be operational.\n")


# --- Client-side Testing for Music Generation Endpoint ---
api_base_url_music = "http://localhost:8000/api/music"

print("\n--- Initiating Real AI Music Generation Test ---")
post_url_music = f"{api_base_url_music}/generate"
example_music_params = {
    "genre": "Rock",
    "mood": "Aggressive",
    "tempo": 140,
    "instrumentation": ["electric guitar", "bass guitar", "drums", "heavy synth"],
    "duration": 180, # This is a hint to MusicGen, actual length depends on max_new_tokens
    "key": "E minor",
    "style_reference": "heavy metal" # Example style reference
}

try:
    response = requests.post(post_url_music, json=example_music_params)
    response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
    init_response = response.json()
    job_id = init_response["job_id"]
    print(f"Music Generation Job Initiated. Job ID: {job_id}")
    print(f"Initial Status: {init_response['status']} - {init_response['message']}")

    print("\n--- Polling for Music Generation Status ---")
    status_check_url = f"{api_base_url_music}/generate/status/{job_id}"
    status = "pending"
    output_url = None

    while status not in ["completed", "failed"]:
        time.sleep(10) # Poll every 10 seconds, as MusicGen takes time
        status_response = requests.get(status_check_url)
        status_response.raise_for_status()
        current_status = status_response.json()
        status = current_status["status"]
        progress = current_status.get("progress", 0)
        message = current_status.get("message", "")
        output_url = current_status.get("output_url")
        print(f"Current Status for Job {job_id}: {status} (Progress: {progress}%) - {message}")
        if status == "completed" and output_url:
            break # Exit loop once completed and URL is available

    print("\n--- Final Music Generation Result ---")
    if status == "completed":
        print(f"Music Generation Completed Successfully!")
        print(f"Output URL: {output_url}")
        print(f"To download and listen to the generated music, use: !wget '{output_url}'")
    else:
        print(f"Music Generation Failed. Reason: {message}")

except requests.exceptions.ConnectionError as e:
    print(f"Error: Could not connect to the API. Ensure the FastAPI app is properly running. {e}")
except requests.exceptions.HTTPError as e:
    print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
except Exception as e:
    print(f"An unexpected error occurred during API test: {e}")
