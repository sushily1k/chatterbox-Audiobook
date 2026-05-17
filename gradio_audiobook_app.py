import os
import random
import tempfile

import numpy as np
import torch
import gradio as gr

from chatterbox.tts import ChatterboxTTS
from audiobook_processor import (
    split_into_chunks,
    parse_multivoice,
    detect_characters,
    combine_audio_chunks,
    make_silence,
    save_audio,
)

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"[Chatterbox Audiobook] Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

MAX_CHUNK_CHARS = 280
SILENCE_SAME_CHAR = 0.15   # seconds between chunks of the same speaker
SILENCE_CHAR_CHANGE = 0.40  # seconds between different speakers


def _set_seed(seed: int):
    if seed != 0:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed % (2**32))


def _combine(chunks, sr):
    """Stack a list of 1-D numpy arrays with silence between them."""
    if not chunks:
        return np.zeros(sr, dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _silence_array(seconds: float, sr: int) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


# ---------------------------------------------------------------------------
# Single-voice generator
# ---------------------------------------------------------------------------

def generate_single(
    script,
    voice_path,
    exaggeration,
    cfg_weight,
    temperature,
    min_p,
    top_p,
    rep_penalty,
    seed,
    model,
):
    if not script or not script.strip():
        yield "**Error:** Please enter a script.", None, gr.update(visible=False)
        return

    if voice_path is None:
        yield "**Error:** Please upload or record a reference voice.", None, gr.update(visible=False)
        return

    if model is None:
        yield "**Error:** Model is still loading — please wait a moment.", None, gr.update(visible=False)
        return

    _set_seed(int(seed))

    chunks = split_into_chunks(script.strip(), max_chars=MAX_CHUNK_CHARS)
    total = len(chunks)
    sr = model.sr
    audio_chunks = []

    yield f"Preparing voice… (0/{total} chunks)", None, gr.update(visible=False)
    try:
        conds = model.prepare_conditionals(voice_path, exaggeration=exaggeration)
    except Exception as exc:
        yield f"**Error** preparing voice: {exc}", None, gr.update(visible=False)
        return

    for i, chunk in enumerate(chunks, 1):
        yield f"Generating chunk {i}/{total}…", None, gr.update(visible=False)
        try:
            wav = model.generate(
                chunk,
                conds,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                min_p=min_p,
                top_p=top_p,
                repetition_penalty=rep_penalty,
            )
            arr = wav.squeeze(0).numpy().astype(np.float32)
        except Exception as exc:
            yield f"**Error** on chunk {i}: {exc}", None, gr.update(visible=False)
            return

        audio_chunks.append(arr)
        if i < total:
            audio_chunks.append(_silence_array(SILENCE_SAME_CHAR, sr))

        combined = _combine(audio_chunks, sr)
        yield (
            f"Chunk {i}/{total} done — streaming preview…",
            (sr, combined),
            gr.update(visible=False),
        )

    # Save final file
    out_path = os.path.join(tempfile.gettempdir(), "audiobook_output.wav")
    final_audio = _combine(audio_chunks, sr)
    try:
        save_audio(final_audio, out_path, sr)
    except Exception:
        import soundfile as sf
        sf.write(out_path, final_audio, sr)

    yield (
        f"**Done!** Generated {total} chunk(s). Download below.",
        (sr, final_audio),
        gr.update(visible=True, value=out_path),
    )


# ---------------------------------------------------------------------------
# Multi-voice generator
# ---------------------------------------------------------------------------

def generate_multivoice(
    script,
    exaggeration,
    cfg_weight,
    temperature,
    min_p,
    top_p,
    rep_penalty,
    char1, v1,
    char2, v2,
    char3, v3,
    char4, v4,
    char5, v5,
    model,
):
    if not script or not script.strip():
        yield "**Error:** Please enter a script.", None, gr.update(visible=False)
        return

    if model is None:
        yield "**Error:** Model is still loading — please wait a moment.", None, gr.update(visible=False)
        return

    # Build voice map from non-empty name/path pairs
    raw_pairs = [
        (char1, v1), (char2, v2), (char3, v3), (char4, v4), (char5, v5),
    ]
    voice_map = {
        name.strip(): path
        for name, path in raw_pairs
        if name and name.strip() and path
    }

    if not voice_map:
        yield "**Error:** Please assign at least one character name and voice file.", None, gr.update(visible=False)
        return

    # Fall-back voice = first entry
    fallback_voice = next(iter(voice_map.values()))

    segments = parse_multivoice(script.strip())
    total_segs = len(segments)
    sr = model.sr
    audio_chunks = []
    prev_char = None
    conds_cache: dict[str, object] = {}  # voice_path → Conditionals

    for seg_idx, (character, text) in enumerate(segments, 1):
        char_label = character or "Narrator"
        voice_path = voice_map.get(char_label, fallback_voice)

        # Encode reference audio once per unique voice path
        if voice_path not in conds_cache:
            yield (
                f"Preparing voice for {char_label}…",
                None,
                gr.update(visible=False),
            )
            try:
                conds_cache[voice_path] = model.prepare_conditionals(
                    voice_path, exaggeration=exaggeration
                )
            except Exception as exc:
                yield (
                    f"**Error** preparing voice for {char_label}: {exc}",
                    None,
                    gr.update(visible=False),
                )
                return
        conds = conds_cache[voice_path]

        text_chunks = split_into_chunks(text.strip(), max_chars=MAX_CHUNK_CHARS)
        n_chunks = len(text_chunks)

        for ci, chunk in enumerate(text_chunks, 1):
            yield (
                f"Segment {seg_idx}/{total_segs} — {char_label} (chunk {ci}/{n_chunks})…",
                None,
                gr.update(visible=False),
            )
            try:
                wav = model.generate(
                    chunk,
                    conds,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                    min_p=min_p,
                    top_p=top_p,
                    repetition_penalty=rep_penalty,
                )
                arr = wav.squeeze(0).numpy().astype(np.float32)
            except Exception as exc:
                yield (
                    f"**Error** on segment {seg_idx} ({char_label}): {exc}",
                    None,
                    gr.update(visible=False),
                )
                return

            # Add silence before appending (not before the very first chunk)
            if audio_chunks:
                if character != prev_char:
                    audio_chunks.append(_silence_array(SILENCE_CHAR_CHANGE, sr))
                else:
                    audio_chunks.append(_silence_array(SILENCE_SAME_CHAR, sr))

            audio_chunks.append(arr)
            prev_char = character

            combined = _combine(audio_chunks, sr)
            yield (
                f"Segment {seg_idx}/{total_segs} — {char_label} chunk {ci}/{n_chunks} done.",
                (sr, combined),
                gr.update(visible=False),
            )

    # Save final file
    out_path = os.path.join(tempfile.gettempdir(), "audiobook_output_multivoice.wav")
    final_audio = _combine(audio_chunks, sr)
    try:
        save_audio(final_audio, out_path, sr)
    except Exception:
        import soundfile as sf
        sf.write(out_path, final_audio, sr)

    yield (
        f"**Done!** Generated {total_segs} segment(s). Download below.",
        (sr, final_audio),
        gr.update(visible=True, value=out_path),
    )


# ---------------------------------------------------------------------------
# Dispatcher (reads mode and delegates)
# ---------------------------------------------------------------------------

def dispatch(
    mode,
    script,
    # single-voice params
    sv_voice, sv_exag, sv_cfg, sv_temp, sv_minp, sv_topp, sv_rep, sv_seed,
    # multi-voice params
    mv_exag, mv_cfg, mv_temp, mv_minp, mv_topp, mv_rep,
    char1, v1, char2, v2, char3, v3, char4, v4, char5, v5,
    model,
):
    if mode == "Single Voice":
        yield from generate_single(
            script, sv_voice, sv_exag, sv_cfg, sv_temp, sv_minp, sv_topp, sv_rep, sv_seed, model
        )
    else:
        yield from generate_multivoice(
            script,
            mv_exag, mv_cfg, mv_temp, mv_minp, mv_topp, mv_rep,
            char1, v1, char2, v2, char3, v3, char4, v4, char5, v5,
            model,
        )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

SCRIPT_PLACEHOLDER = (
    "Enter your audiobook script here.\n\n"
    "For single voice, just paste plain text.\n\n"
    "For multi-voice, tag each line:\n"
    "[Narrator]: It was a dark and stormy night.\n"
    "[Alice]: I'm not afraid of the dark.\n"
    "[Bob]: You should be."
)

with gr.Blocks(title="Chatterbox Audiobook", theme=gr.themes.Soft()) as demo:
    model_state = gr.State(None)

    gr.Markdown("# Chatterbox Audiobook Generator")
    gr.Markdown(
        "Generate audiobooks with realistic voices using Chatterbox TTS. "
        f"Running on **{DEVICE.upper()}**."
    )

    with gr.Row(equal_height=False):

        # ------------------------------------------------------------------ #
        # LEFT COLUMN — inputs                                                #
        # ------------------------------------------------------------------ #
        with gr.Column(scale=1):

            script = gr.Textbox(
                label="Script",
                lines=12,
                max_lines=40,
                placeholder=SCRIPT_PLACEHOLDER,
            )

            mode = gr.Radio(
                label="Mode",
                choices=["Single Voice", "Multi-Voice"],
                value="Single Voice",
            )

            # ---- Single Voice panel ---- #
            with gr.Group(visible=True) as single_panel:
                gr.Markdown("### Single Voice Settings")
                sv_voice = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Reference Voice",
                )
                sv_exag = gr.Slider(
                    minimum=0.25, maximum=2.0, step=0.05, value=0.5,
                    label="Exaggeration",
                )
                sv_cfg = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.05, value=0.5,
                    label="CFG / Pace",
                )
                with gr.Accordion("Advanced", open=False):
                    sv_temp = gr.Slider(
                        minimum=0.05, maximum=5.0, step=0.05, value=0.8,
                        label="Temperature",
                    )
                    sv_minp = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01, value=0.05,
                        label="Min-P",
                    )
                    sv_topp = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01, value=1.0,
                        label="Top-P",
                    )
                    sv_rep = gr.Slider(
                        minimum=1.0, maximum=2.0, step=0.1, value=1.2,
                        label="Repetition Penalty",
                    )
                    sv_seed = gr.Number(value=0, label="Seed (0 = random)", precision=0)

            # ---- Multi-Voice panel ---- #
            with gr.Group(visible=False) as multi_panel:
                gr.Markdown("### Multi-Voice Settings")
                gr.Markdown(
                    "Tag each character as `[Name]: their dialogue`. "
                    "Untagged lines go to **Narrator**."
                )

                detected_chars = gr.Textbox(
                    label="Detected Characters",
                    interactive=False,
                    placeholder="Characters found in your script will appear here…",
                )

                char_names = []
                char_voices = []
                for i in range(1, 6):
                    with gr.Row():
                        cn = gr.Textbox(
                            label=f"Character {i} Name",
                            scale=1,
                            placeholder=f"e.g. {'Narrator' if i == 1 else 'Character ' + str(i)}",
                        )
                        cv = gr.Audio(
                            sources=["upload", "microphone"],
                            type="filepath",
                            label=f"Voice {i}",
                            scale=2,
                        )
                        char_names.append(cn)
                        char_voices.append(cv)

                mv_exag = gr.Slider(
                    minimum=0.25, maximum=2.0, step=0.05, value=0.5,
                    label="Exaggeration",
                )
                mv_cfg = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.05, value=0.5,
                    label="CFG / Pace",
                )
                with gr.Accordion("Advanced", open=False):
                    mv_temp = gr.Slider(
                        minimum=0.05, maximum=5.0, step=0.05, value=0.8,
                        label="Temperature",
                    )
                    mv_minp = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01, value=0.05,
                        label="Min-P",
                    )
                    mv_topp = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01, value=1.0,
                        label="Top-P",
                    )
                    mv_rep = gr.Slider(
                        minimum=1.0, maximum=2.0, step=0.1, value=1.2,
                        label="Repetition Penalty",
                    )

            gen_btn = gr.Button("Generate Audiobook", variant="primary")
            stop_btn = gr.Button("Stop", variant="stop", visible=False)

        # ------------------------------------------------------------------ #
        # RIGHT COLUMN — output                                               #
        # ------------------------------------------------------------------ #
        with gr.Column(scale=1):
            status_md = gr.Markdown("Ready.")
            audio_out = gr.Audio(
                label="Output (updates as chunks complete)",
                interactive=False,
            )
            download_file = gr.File(
                label="Download Final Audiobook",
                visible=False,
            )

    # ------------------------------------------------------------------ #
    # Event wiring                                                         #
    # ------------------------------------------------------------------ #

    # Mode toggle
    mode.change(
        fn=lambda m: (
            gr.update(visible=m == "Single Voice"),
            gr.update(visible=m == "Multi-Voice"),
        ),
        inputs=mode,
        outputs=[single_panel, multi_panel],
    )

    # Auto-detect characters
    script.change(
        fn=lambda t: ", ".join(detect_characters(t)) if t and t.strip() else "",
        inputs=script,
        outputs=detected_chars,
    )

    # Show/hide stop button during generation
    def _show_stop():
        return gr.update(visible=True)

    def _hide_stop():
        return gr.update(visible=False)

    all_inputs = [
        mode,
        script,
        sv_voice, sv_exag, sv_cfg, sv_temp, sv_minp, sv_topp, sv_rep, sv_seed,
        mv_exag, mv_cfg, mv_temp, mv_minp, mv_topp, mv_rep,
        char_names[0], char_voices[0],
        char_names[1], char_voices[1],
        char_names[2], char_voices[2],
        char_names[3], char_voices[3],
        char_names[4], char_voices[4],
        model_state,
    ]

    gen_event = gen_btn.click(
        fn=dispatch,
        inputs=all_inputs,
        outputs=[status_md, audio_out, download_file],
    )

    # Show stop button when generation starts, hide when done
    gen_btn.click(fn=_show_stop, inputs=[], outputs=[stop_btn])
    gen_event.then(fn=_hide_stop, inputs=[], outputs=[stop_btn])

    # Stop button cancels generation
    stop_btn.click(fn=None, cancels=[gen_event])
    stop_btn.click(fn=_hide_stop, inputs=[], outputs=[stop_btn])

    # Load model on app start
    def _load_model():
        print(f"[Chatterbox] Loading model on {DEVICE}…")
        model = ChatterboxTTS.from_pretrained(device=DEVICE)
        if DEVICE == "mps":
            # S3Tokenizer has layers with >65536 channels that MPS cannot run.
            # It is only used in prepare_conditionals (never during generation),
            # so keeping it on CPU is free — generation stays fully on MPS.
            model.s3gen.tokenizer.to("cpu")
            print("[Chatterbox] s3tokenizer pinned to CPU (MPS channel limit workaround)")
        return model

    demo.load(fn=_load_model, outputs=model_state)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
