import os
import sys
import whisper
from pyannote.audio import Pipeline
from loader import Loader

AUDIO_FILE_PATH = "src/asset/sample_audio_1.mp3"
AUDIO_FILE_LANGUAGE = "en"
HF_TOKEN = "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"

def transcribe_audio(file_path, language):
    print("\nStarting transcription...")
    loader = Loader(message="Transcribing audio... ")
    loader.start()
    
    # Load Whisper model
    model = whisper.load_model("large")
    
    # Transcribe the audio file into segments
    result = model.transcribe(file_path, language=language, task="transcribe", word_timestamps=True)
    print("Transcription raw result", result)
    segments = result['segments']
    
    loader.stop()
    print("\nTranscription complete!")
    
    return segments

def speaker_diarization(file_path):
    print("\nStarting speaker diarization...")
    
    loader = Loader(message="Speaker diarization in progress... ")
    loader.start()
    
    # Load the speaker diarization pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_TOKEN)
    diarization = pipeline({"uri": file_path, "audio": file_path})

    # Parse result
    speaker_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append((turn.start, turn.end, speaker))
        
    loader.stop()
    print("\nSpeaker diarization complete!")
    
    return speaker_segments

def format_conversation(transcription, speaker_segments):
    formatted_conversation = []
    speaker_index = 0
    
    # Iterate over segments and assign them to speakers based on timing
    for segment in transcription:
        segment_start = segment['start']
        segment_text = segment['text']
        
        # Find the matching speaker segment for this word based on start time
        while (speaker_index < len(speaker_segments) and 
               speaker_segments[speaker_index][1] < segment_start):
            speaker_index += 1

        if speaker_index < len(speaker_segments):
            start, end, speaker = speaker_segments[speaker_index]
            
            # Check if the word timestamp falls within the speaker segment
            if start <= segment_start <= end:
                speaker_id = f"Speaker {speaker}"
                formatted_conversation.append({
                    "speaker": speaker_id,
                    "text": segment_text
                })

    # Combine consecutive words for the same speaker
    combined_conversation = []
    current_speaker = None
    current_text = []

    for entry in formatted_conversation:
        if entry['speaker'] == current_speaker:
            current_text.append(entry['text'])
        else:
            if current_text:
                combined_conversation.append({
                    "speaker": current_speaker,
                    "text": " ".join(current_text)
                })
            current_speaker = entry['speaker']
            current_text = [entry['text']]

    # Add the last accumulated text
    if current_text:
        combined_conversation.append({
            "speaker": current_speaker,
            "text": " ".join(current_text)
        })

    return combined_conversation

def main():
    if not os.path.isfile(AUDIO_FILE_PATH):
        print("File not found.")
        sys.exit(1)

    # Transcribe audio
    transcription = transcribe_audio(AUDIO_FILE_PATH, AUDIO_FILE_LANGUAGE)
    print("Transcription:", transcription)
    
    # Speaker Diarization
    speaker_segments = speaker_diarization(AUDIO_FILE_PATH)
    print("Speaker Segments:", speaker_segments)

    # Format structured conversation
    structured_conversation = format_conversation(transcription, speaker_segments)
    
    # Show results
    for entry in structured_conversation:
        print(f"{entry['speaker']}: {entry['text']}")

if __name__ == "__main__":
    main()
