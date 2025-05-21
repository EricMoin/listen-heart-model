import shutil
import tempfile
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
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

    tmp_input_audio_path = None
    generated_response_audio_path = None

    try:
        # Create a temporary file to save the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_input_audio_path = tmp.name

        print(f"Processing uploaded audio file: {tmp_input_audio_path}")
        # Process the audio file. This method is expected to return the path to the generated response audio file.
        # Or an error string if TTS fails.
        generated_response_audio_path = assistant.process_audio(
            tmp_input_audio_path)

        # Check if process_audio returned a valid path or an error string
        if not generated_response_audio_path or not isinstance(generated_response_audio_path, str) or not os.path.exists(generated_response_audio_path):
            # This could be an error message from process_audio, or an invalid path
            error_detail = generated_response_audio_path if isinstance(
                generated_response_audio_path, str) else "Failed to generate response audio or invalid path returned."
            print(f"Error from process_audio or invalid path: {error_detail}")
            # Check if the path points to a directory that VoiceMoodTreeHole tries to create for its output
            # self.output_dir = os.path.join(base_dir, "output") # from main.py
            # If the output dir itself is what's returned, that's an error
            if generated_response_audio_path == assistant.output_dir:
                error_detail = "TTS process failed to generate a specific audio file, returned output directory instead."

            # Check if the returned path is actually the input path (which means TTS might have failed silently or was skipped)
            if generated_response_audio_path == tmp_input_audio_path:
                error_detail = "TTS process did not generate a new audio file; got input path back."

            # Check specific error message from `main.py` if TTS fails
            if "生成语音回应时出错" in error_detail:
                raise HTTPException(
                    status_code=500, detail=f"Error during Text-to-Speech synthesis: {error_detail}")

            raise HTTPException(status_code=500, detail=error_detail)

        # Return the generated audio file
        # The TTS in main.py saves as WAV by default in text_to_speech.py
        # The file name generated by process_audio already includes the .wav extension.
        response_filename = os.path.basename(generated_response_audio_path)
        return FileResponse(path=generated_response_audio_path, media_type="audio/wav", filename=response_filename)

    except FileNotFoundError as e:
        print(f"FileNotFoundError during processing: {e}")
        raise HTTPException(
            status_code=500, detail=f"A required file was not found: {e}")
    except HTTPException:  # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        print(f"Error processing audio: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"An internal error occurred: {str(e)}")
    finally:
        # Clean up the temporary uploaded input file
        if tmp_input_audio_path and os.path.exists(tmp_input_audio_path):
            os.remove(tmp_input_audio_path)
        # The generated_response_audio_path is in the 'output' directory and might be intended to persist for a bit,
        # or be cleaned up by a separate mechanism. If it needs to be cleaned here, add:
        # if generated_response_audio_path and os.path.exists(generated_response_audio_path):
        #     os.remove(generated_response_audio_path)

        # Close the uploaded file object
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
