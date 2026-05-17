import re
import numpy as np
import torch
import torchaudio
from pathlib import Path


def split_into_chunks(text: str, max_chars: int = 280) -> list[str]:
    sentence_pattern = re.compile(r'[^.!?]*[.!?]+')
    chunks = []

    def split_long(segment: str) -> list[str]:
        if len(segment) <= max_chars:
            return [segment]
        parts = re.split(r'(?<=[,—])\s*', segment)
        result = []
        current = ""
        for part in parts:
            if not current:
                current = part
            elif len(current) + 1 + len(part) <= max_chars:
                current = current + " " + part
            else:
                if len(current) > max_chars:
                    words = current.split()
                    word_buf = ""
                    for word in words:
                        if not word_buf:
                            word_buf = word
                        elif len(word_buf) + 1 + len(word) <= max_chars:
                            word_buf = word_buf + " " + word
                        else:
                            result.append(word_buf)
                            word_buf = word
                    if word_buf:
                        result.append(word_buf)
                else:
                    result.append(current)
                current = part
        if current:
            if len(current) > max_chars:
                words = current.split()
                word_buf = ""
                for word in words:
                    if not word_buf:
                        word_buf = word
                    elif len(word_buf) + 1 + len(word) <= max_chars:
                        word_buf = word_buf + " " + word
                    else:
                        result.append(word_buf)
                        word_buf = word
                if word_buf:
                    result.append(word_buf)
            else:
                result.append(current)
        return result

    sentences = []
    last_end = 0
    for match in sentence_pattern.finditer(text):
        sentences.append(match.group())
        last_end = match.end()
    remainder = text[last_end:].strip()
    if remainder:
        sentences.append(remainder)

    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if not current_chunk:
            if len(sentence) > max_chars:
                for part in split_long(sentence):
                    part = part.strip()
                    if part:
                        chunks.append(part)
            else:
                current_chunk = sentence
        elif len(current_chunk) + 1 + len(sentence) <= max_chars:
            current_chunk = current_chunk + " " + sentence
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            if len(sentence) > max_chars:
                for part in split_long(sentence):
                    part = part.strip()
                    if part:
                        chunks.append(part)
                current_chunk = ""
            else:
                current_chunk = sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return [c for c in chunks if c]


def parse_multivoice(text: str) -> list[tuple[str, str]]:
    tag_pattern = re.compile(r'^\[([^\]]+)\]:\s*', re.IGNORECASE)
    lines = text.splitlines()
    result = []
    current_character = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        match = tag_pattern.match(stripped)
        if match:
            current_character = match.group(1).strip()
            dialogue = stripped[match.end():].strip()
            if dialogue:
                result.append((current_character, dialogue))
        else:
            if current_character is None:
                current_character = "Narrator"
            result.append((current_character, stripped))

    return result


def detect_characters(text: str) -> list[str]:
    tag_pattern = re.compile(r'^\[([^\]]+)\]:\s*', re.IGNORECASE)
    lines = text.splitlines()
    characters = set()
    has_untagged = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        match = tag_pattern.match(stripped)
        if match:
            characters.add(match.group(1).strip())
        else:
            has_untagged = True

    if has_untagged or not characters:
        characters.add("Narrator")

    return sorted(characters)


def make_silence(duration_secs: float, sample_rate: int = 24000) -> np.ndarray:
    length = int(duration_secs * sample_rate)
    return np.zeros(length, dtype=np.float32)


def combine_audio_chunks(
    chunks: list[np.ndarray],
    silence_secs: float = 0.15,
    sample_rate: int = 24000,
) -> np.ndarray:
    if not chunks:
        return np.array([], dtype=np.float32)
    silence = make_silence(silence_secs, sample_rate)
    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(chunk.astype(np.float32))
        if i < len(chunks) - 1:
            parts.append(silence)
    return np.concatenate(parts, axis=0)


def save_audio(audio: np.ndarray, path: str, sample_rate: int = 24000) -> None:
    tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    torchaudio.save(path, tensor, sample_rate)
