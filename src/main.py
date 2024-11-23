import os
import sys
import uuid
import faster_whisper
import torch
import torchaudio
import ollama
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

from helpers import (
    deleteFileOrDir,
    create_config,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    write_srt,
)
from prompt import QNA_PROMPT_MESSAGE

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InterviewAnalysisRequest(BaseModel):
    audio_url: str

LLM = "llama3"
TEMPERATURE = 0.2

def create_prompt(transcription: str) -> str:
    return QNA_PROMPT_MESSAGE.replace("<TRANSCRIPTION>", transcription)

# Configuration
COMPUTING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_MODEL = "medium.en"
WHISPER_BATCH_SIZE = 8
mtypes = {"cpu": "int8", "cuda": "float16"}
language = None

@app.post("/analyse_interview")
async def analyse_error(request: InterviewAnalysisRequest):
    # Generate unique session directory
    session_id = str(uuid.uuid4())
    session_path = os.path.join("src/temp_outputs", session_id)
    os.makedirs(session_path, exist_ok=True)

    # AUDIO_FILE_PATH = request.audio_url

    # if not os.path.exists(AUDIO_FILE_PATH):
    #     sys.exit(f"Audio file does not exist. Path provided: {AUDIO_FILE_PATH}")

    # Transcribe the audio file
    whisper_model = faster_whisper.WhisperModel(
        WHISPER_MODEL,
        device=COMPUTING_DEVICE,
        compute_type=mtypes[COMPUTING_DEVICE]
    )
    whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
    audio_waveform = faster_whisper.decode_audio(request.audio_url)
    audio_waveform_tensors = torch.from_numpy(audio_waveform)

    transcript_segments, transcript_info = whisper_pipeline.transcribe(
        audio_waveform,
        batch_size=WHISPER_BATCH_SIZE
    )

    full_transcript = "".join(segment.text for segment in transcript_segments)
    
    # return {"unprocessed transcript": full_transcript}

    # clear gpu vram
    # del whisper_model, whisper_pipeline
    # torch.cuda.empty_cache()

    # Forced Alignment
    alignment_model, alignment_tokenizer = load_alignment_model(
        COMPUTING_DEVICE,
        dtype=torch.float16 if COMPUTING_DEVICE == "cuda" else torch.float32,
    )

    emissions, stride = generate_emissions(
        alignment_model,
        audio_waveform_tensors.to(alignment_model.dtype).to(alignment_model.device),
        batch_size=WHISPER_BATCH_SIZE
    )

    # clear gpu vram
    # del alignment_model
    # torch.cuda.empty_cache()

    tokens_starred, text_starred = preprocess_text(
        full_transcript,
        romanize=True,
        language=langs_to_iso[transcript_info.language],
    )

    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        alignment_tokenizer,
    )

    spans = get_spans(tokens_starred, segments, blank_token)

    word_timestamps = postprocess_results(text_starred, spans, stride, scores)

    # Convert audio to mono for NeMo compatibility
    mono_audio_path = os.path.join(session_path, "mono_file.wav")
    torchaudio.save(
        mono_audio_path,
        audio_waveform_tensors.cpu().unsqueeze(0).float(),
        16000
    )

    # Initialize NeMo MSDD diarization model
    msdd_model = NeuralDiarizer(cfg=create_config(session_path)).to(COMPUTING_DEVICE)
    msdd_model.diarize()

    # clear gpu vram
    del msdd_model
    torch.cuda.empty_cache()

    # Reading timestamps <> Speaker Labels mapping
    speaker_ts = []
    rttm_path = os.path.join(session_path, "pred_rttms", "mono_file.rttm")
    with open(rttm_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

    # Write output files in session-specific paths
    # transcript_txt_path = os.path.join(session_path, f"{session_id}.txt")
    # transcript_srt_path = os.path.join(session_path, f"{session_id}.srt")
    # with open(transcript_txt_path, "w", encoding="utf-8-sig") as f:
    #     get_speaker_aware_transcript(ssm, f)
    # with open(transcript_srt_path, "w", encoding="utf-8-sig") as srt:
    #     write_srt(ssm, srt)
        
    processed_transcript = get_speaker_aware_transcript(ssm)
    
    # return {"transcript": processed_transcript}

    # Cleanup after processing
    # deleteFileOrDir(session_path)
        
    prompt = create_prompt(processed_transcript)
            
    response = ollama.chat(
        model=LLM,
        options={"temperature": TEMPERATURE},
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = response.get("message", {}).get("content", "")
    cleaned_response_text = response_text.strip().replace('\\n', ' ').replace('\\"', '"')

    # try:
    #     response_json = json.loads(cleaned_response_text)
    #     if not all(key in response_json for key in ["cause", "impact", "fix"]):
    #         raise ValueError("Invalid response format")
    # except (json.JSONDecodeError, ValueError) as e:
    #     print(f"Failed to parse response: {cleaned_response_text}")
    #     raise HTTPException(status_code=500, detail=f"Invalid response format: {str(e)}")

    return {"analysis": cleaned_response_text}