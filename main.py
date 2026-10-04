import argparse
from dataclasses import dataclass
import os

from anthropic import Anthropic
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

@dataclass
class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3"
    API_KEY=os.environ.get("ANTHROPIC_API_KEY")
    ANTHROPIC_WORKSPACE=os.environ.get("ANTHROPIC_WORKSPACE"),
    language = "French" # the default
    level = "B2" # the default

cfg = Config()

def get_questions(transcript: str) -> str:


    with open("question-prompt.md", "r") as f:
        prompt = f.read()

    prompt = prompt.replace("{LANGUAGE}", cfg.language)
    prompt = prompt.replace("{LANGUAGE_LEVEL}", cfg.level)
    prompt = prompt.replace("{TRANSCRIPTION}", transcript)

    client = Anthropic(
        api_key=cfg.API_KEY,
    )

    message = client.messages.create(
        max_tokens=10_000,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="claude-opus-5-5",
        extra_headers={"anthropic-workspace-id": cfg.ANTHROPIC_WORKSPACE},
    )

    responses = []
    for block in message.content:
        if block.type == "text":
            responses.append(block.text)

    return " ".join(responses)


def main(args):
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
            cfg.model_id, dtype=cfg.dtype, low_cpu_mem_usage=True, use_safetensors=True,
    )

    model.to(cfg.device)

    processor = AutoProcessor.from_pretrained(cfg.model_id)

    # no default language as this is 
    pipe = pipeline(
            "automatic-speech-recognition",
            model=cfg.model_id,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=cfg.dtype,
            chunk_length_s=30,
            device=cfg.device,
    )

    results = pipe(args.filename, batch_size=4, return_timestamps=True)["chunks"]

    total_text = " ".join([result["text"] for result in results])

    claude_response = get_questions(total_text)

    with open(f"{args.filename}.txt", "w") as f:
        f.write(total_text)

    with open(f"{args.filename}_questions.txt", "w") as f:
        f.write(claude_response)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="Audio Transcriber", description="Transcriber for any audio")
    parser.add_argument("filename")
    args = parser.parse_args()

    main(args)
