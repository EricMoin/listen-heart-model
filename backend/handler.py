import shutil
import tempfile
import os
import time
import subprocess
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import sys

# Add the main directory to the Python path
# current_handler_dir is 'backend/'
current_handler_dir = os.path.dirname(os.path.abspath(__file__))
# project_root is 'listen-heart-model/'
project_root = os.path.dirname(current_handler_dir)
# main_code_dir is 'listen-heart-model/main/'
main_code_dir = os.path.join(project_root, "main")

# Add the 'main' directory itself to sys.path so that main.py and its sibling modules
# (like audio_preprocessing.py) can be imported directly.
if main_code_dir not in sys.path:
    sys.path.insert(0, main_code_dir)

# Import the main class
# VoiceMoodTreeHole is in main.py, which is in main_code_dir.
# Since main_code_dir is in sys.path, main.py is importable as 'main'.
try:
    from main import VoiceMoodTreeHole
except ImportError as e:
    print(
        f"Error importing VoiceMoodTreeHole from 'main.py' in '{main_code_dir}': {e}")
    print(f"Current sys.path: {sys.path}")
    # As a fallback, ensure project_root is also in path and try the old import style
    # This might indicate a more complex packaging or __init__.py situation in 'main'
    if project_root not in sys.path:
        sys.path.insert(0, project_root)  # Ensure project root is also in path

    print(f"Retrying import with 'from main.main import VoiceMoodTreeHole'...")
    try:
        from main.main import VoiceMoodTreeHole
        print("Fallback import 'from main.main import VoiceMoodTreeHole' succeeded.")
    except ImportError as e2:
        print(f"Fallback import also failed: {e2}")
        print("Please check the Python path and ensure 'main/main.py' exists and is correctly structured.")
        raise e  # Raise the original error if fallback also fails


app = FastAPI()

# Configure CORS
origins = [
    "http://localhost",       # Allow localhost for development
    "http://localhost:3000",  # Allow your frontend running on port 3000
    # You can add other origins here, like your deployed frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of origins that are allowed to make cross-origin requests
    allow_credentials=True,  # Allow cookies to be included in cross-origin requests
    allow_methods=["*"],    # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],    # Allow all headers
)

# Global variable to hold the model instance
assistant = None


@app.on_event("startup")
async def startup_event():
    global assistant
    print("Loading models...")
    try:
        # Load models. This class method loads them into class variables.
        VoiceMoodTreeHole.load_models()
        # Create an instance of the assistant
        assistant = VoiceMoodTreeHole()
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models during startup: {e}")
        assistant = None


@app.post("/analyze_audio/")
async def analyze_audio(file: UploadFile = File(...)):
    global assistant
    if assistant is None or not VoiceMoodTreeHole._models_loaded:
        raise HTTPException(
            status_code=503, detail="Models are not loaded or failed to load. Please check server logs.")

    tmp_uploaded_audio_path = None
    tmp_wav_audio_path = None
    generated_response_audio_path = None

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tmp_dir = os.path.join(script_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        original_extension = os.path.splitext(file.filename)[1]
        tmp_uploaded_filename = f"input_{int(time.time())}{original_extension}"
        tmp_uploaded_audio_path = os.path.join(tmp_dir, tmp_uploaded_filename)

        with open(tmp_uploaded_audio_path, "wb") as tmp_file_obj:
            shutil.copyfileobj(file.file, tmp_file_obj)
        # print(f"Uploaded audio file saved: {tmp_uploaded_audio_path}") # Less verbose logging

        base_filename = os.path.splitext(tmp_uploaded_filename)[0]
        tmp_wav_filename = f"{base_filename}.wav"
        tmp_wav_audio_path = os.path.join(tmp_dir, tmp_wav_filename)

        ffmpeg_command = [
            "ffmpeg", "-y",  # -y overwrites output file without asking
            "-i", tmp_uploaded_audio_path,
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            tmp_wav_audio_path
        ]

        try:
            # print(f"Attempting to convert to WAV: {' '.join(ffmpeg_command)}")
            process = subprocess.run(
                # Added encoding
                ffmpeg_command, check=True, capture_output=True, text=True, encoding='utf-8')
            # print(f"FFmpeg STDOUT: {process.stdout}") # Usually not needed if successful
            if process.stderr:
                print(f"FFmpeg STDERR (Info/Warnings): {process.stderr}")
            # print(f"Converted WAV file saved: {tmp_wav_audio_path}")
        except FileNotFoundError:
            print(
                f"Error: ffmpeg command not found. Ensure ffmpeg is installed and in PATH.")
            raise HTTPException(
                status_code=500, detail="Audio conversion tool (ffmpeg) not found.")
        except subprocess.CalledProcessError as e:
            print(
                f"Error during ffmpeg conversion for {tmp_uploaded_audio_path} to {tmp_wav_audio_path}. FFmpeg stderr: {e.stderr}")
            raise HTTPException(
                status_code=500, detail=f"Failed to convert audio to WAV. Error: {e.stderr}")

        # print(f"Processing converted WAV file: {tmp_wav_audio_path}")
        generated_response_audio_path = assistant.process_audio(
            tmp_wav_audio_path)

        if not generated_response_audio_path or not isinstance(generated_response_audio_path, str) or not os.path.exists(generated_response_audio_path):
            error_detail = generated_response_audio_path if isinstance(
                generated_response_audio_path, str) else "Process_audio failed or returned invalid path."

            # Consolidate specific checks for error_detail
            if generated_response_audio_path == assistant.output_dir:
                error_detail = "TTS process returned output directory instead of a file."
            elif generated_response_audio_path == tmp_wav_audio_path:  # Check against WAV path now
                error_detail = "TTS process did not generate a new audio file; got converted input path back."
            # Ensure error_detail is string for 'in'
            elif "生成语音回应时出错" in str(error_detail):
                error_detail = f"Error during Text-to-Speech synthesis: {error_detail}"

            print(
                f"Error after process_audio: {error_detail}. Input WAV: {tmp_wav_audio_path}")
            raise HTTPException(status_code=500, detail=error_detail)

        response_filename = os.path.basename(generated_response_audio_path)
        return FileResponse(path=generated_response_audio_path, media_type="audio/wav", filename=response_filename)

    except FileNotFoundError as e:  # This might catch issues if tmp_dir creation fails, though unlikely with exist_ok=True
        print(f"FileNotFoundError during processing: {e}")
        raise HTTPException(
            status_code=500, detail=f"A required file or directory was not found: {e}")
    except HTTPException:
        raise  # Re-raise HTTPExceptions directly
    except Exception as e:
        print(f"Unexpected error processing audio: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"An internal server error occurred: {str(e)}")
    finally:
        if tmp_uploaded_audio_path and os.path.exists(tmp_uploaded_audio_path):
            try:
                os.remove(tmp_uploaded_audio_path)
            except Exception as e:
                print(
                    f"Error deleting temporary uploaded file {tmp_uploaded_audio_path}: {e}")
        if tmp_wav_audio_path and os.path.exists(tmp_wav_audio_path):
            try:
                os.remove(tmp_wav_audio_path)
            except Exception as e:
                print(
                    f"Error deleting temporary WAV file {tmp_wav_audio_path}: {e}")
        if file:  # Ensure file object exists before trying to close
            await file.close()


@app.get("/")
async def root():
    return {"message": "VoiceMoodTreeHole API is running. Use /analyze_audio/ to process audio and get audio response."}

if __name__ == "__main__":
    import uvicorn
    # The sys.path modification at the top of the file ensures that 'main' module
    # and its internal imports work correctly when running this handler directly.

    print("Starting Uvicorn server for handler.py...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"sys.path includes: {main_code_dir}")
    print("Make sure the 'models' directory is correctly placed relative to the 'main' directory (e.g., '../models' from 'main/main.py').")

    uvicorn.run(app, host="0.0.0.0", port=8000)
