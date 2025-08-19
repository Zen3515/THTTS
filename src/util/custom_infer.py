# src: https://github.com/VYNCX/F5-TTS-THAI/blob/99b8314f66a14fc2f0a6b53e5122829fbdf9c59c/src/f5_tts/infer/utils_infer.py

import re
import tqdm
import torch
import numpy as np
import syllapy
import torchaudio

from ssg import syllable_tokenize
from concurrent.futures import ThreadPoolExecutor
from f5_tts.model.utils import convert_char_to_pinyin
from f5_tts.infer.utils_infer import target_sample_rate, hop_length, mel_spec_type, target_rms, cross_fade_duration, nfe_step, cfg_strength, sway_sampling_coef, speed, fix_duration, device


from util.ipa import any_ipa


def custom_chunk_text(text: str, max_chars=200):
    """
    Splits the input text into chunks by breaking at spaces, creating visually balanced chunks.

    Args:
        text (str): The text to be split.
        max_chars (int): Approximate maximum number of bytes per chunk in UTF-8 encoding.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks: list[str] = []
    current_chunk = ""
    # Replace spaces with <unk> if desired, then split on <unk> or spaces
    text = text.replace(" ", "<unk>")
    segments = re.split(r"(<unk>|\s+)", text)

    for segment in segments:
        if not segment or segment in ("<unk>", " "):
            continue
        # Check the byte length for UTF-8 encoding
        if len((current_chunk + segment).encode("utf-8")) <= max_chars:
            current_chunk += segment
            current_chunk += " "  # Add space after each segment for readability
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = segment + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Replace <unk> back with spaces in the final output
    chunks = [chunk.replace("<unk>", " ") for chunk in chunks]

    return chunks


# estimated duration with syllable
FRAMES_PER_SEC = target_sample_rate / hop_length


def words_to_frame(text: str, frame_per_words: int):
    thai_pattern = r'[\u0E00-\u0E7F\s]+'
    english_pattern = r'[a-zA-Z\s]+'

    thai_segs = re.findall(thai_pattern, text)
    eng_segs = re.findall(english_pattern, text)

    syl_th = sum(len(syllable_tokenize(seg.strip())) for seg in thai_segs if seg.strip())
    syl_en = sum(syllapy._syllables(seg.strip()) for seg in eng_segs if seg.strip())
    syl_unk = text.count(',')  # Count spaces as 1 syllable each

    duration = (syl_th + syl_en + syl_unk) * frame_per_words
    # print(f"Thai: {syl_th}, Eng: {syl_en}, Spaces: {syl_unk}, Total: {duration} frames")
    return duration


def custom_infer_process(
    ref_audio,
    ref_text,
    gen_text,
    model_obj,
    vocoder,
    mel_spec_type=mel_spec_type,
    show_info=print,
    progress=tqdm,
    target_rms=target_rms,
    cross_fade_duration=cross_fade_duration,
    nfe_step=nfe_step,
    cfg_strength=cfg_strength,
    sway_sampling_coef=sway_sampling_coef,
    speed=speed,
    fix_duration=fix_duration,
    device=device,
    set_max_chars=250,
    use_ipa=False
):
    # Split the input text into batches
    audio, sr = torchaudio.load(ref_audio)
    # max_chars = int(len(ref_text.encode("utf-8")) / (audio.shape[-1] / sr) * (22 - audio.shape[-1] / sr) * speed)
    gen_text_batches = custom_chunk_text(gen_text, max_chars=set_max_chars)
    for i, gen_text in enumerate(gen_text_batches):
        print(f"gen_text {i}", gen_text)
    print("\n")

    show_info(f"Generating audio in {len(gen_text_batches)} batches...")
    return next(
        custom_infer_batch_process(
            (audio, sr),
            ref_text,
            gen_text_batches,
            model_obj,
            vocoder,
            mel_spec_type=mel_spec_type,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=device,
            use_ipa=use_ipa
        )
    )

# infer batches


def custom_infer_batch_process(
    ref_audio,
    ref_text,
    gen_text_batches,
    model_obj,
    vocoder,
    mel_spec_type="vocos",
    progress=tqdm,
    target_rms=0.1,
    cross_fade_duration=0.15,
    nfe_step=32,
    cfg_strength=2.0,
    sway_sampling_coef: float = -1,
    speed: float = 1,
    fix_duration=None,
    device=None,
    streaming=False,
    chunk_size=2048,
    use_ipa=False
):
    audio, sr = ref_audio
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    audio = audio.to(device)

    generated_waves = []
    spectrograms = []

    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "

    def process_batch(gen_text):
        local_speed = speed
        if len(gen_text.encode("utf-8")) < 15:
            local_speed = 0.3

        # Prepare the text
        if use_ipa:
            ref_text_ipa = any_ipa(ref_text)
            gen_text_ipa = any_ipa(gen_text)
            final_text_list = [ref_text_ipa + " " + gen_text_ipa]  # pyright: ignore[reportOperatorIssue]
        else:
            text_list = [ref_text + gen_text]
            final_text_list = convert_char_to_pinyin(text_list)

        ref_audio_len = audio.shape[-1] // hop_length
        if fix_duration is not None:
            duration = int(fix_duration * target_sample_rate / hop_length)
        else:
            # Calculate duration
            FRAMES_PER_WORDS = FRAMES_PER_SEC / 4
            speech_rate = int(FRAMES_PER_WORDS / local_speed)
            duration = ref_audio_len + words_to_frame(text=gen_text, frame_per_words=speech_rate)

        # inference
        with torch.inference_mode():
            generated, _ = model_obj.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                lens=torch.tensor([ref_audio_len], device=device, dtype=torch.long)
            )
            del _

            generated = generated.to(torch.float32)  # generated mel spectrogram
            generated = generated[:, ref_audio_len:, :]
            generated = generated.permute(0, 2, 1)
            if mel_spec_type == "vocos":
                generated_wave = vocoder.decode(generated)
            elif mel_spec_type == "bigvgan":
                generated_wave = vocoder(generated)
            if rms < target_rms:
                generated_wave = generated_wave * rms / target_rms

            # wav -> numpy
            generated_wave = generated_wave.squeeze().cpu().numpy()

            if streaming:
                for j in range(0, len(generated_wave), chunk_size):
                    yield generated_wave[j: j + chunk_size], target_sample_rate
            else:
                generated_cpu = generated[0].cpu().numpy()
                del generated
                yield generated_wave, generated_cpu

    if streaming:
        for gen_text in progress.tqdm(gen_text_batches) if progress is not None else gen_text_batches:
            for chunk in process_batch(gen_text):
                yield chunk
    else:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_batch, gen_text) for gen_text in gen_text_batches]
            for future in progress.tqdm(futures) if progress is not None else futures:
                result = future.result()
                if result:
                    generated_wave, generated_mel_spec = next(result)
                    generated_waves.append(generated_wave)
                    spectrograms.append(generated_mel_spec)

        if generated_waves:
            if cross_fade_duration <= 0:
                # Simply concatenate
                final_wave = np.concatenate(generated_waves)
            else:
                # Combine all generated waves with cross-fading
                final_wave = generated_waves[0]
                for i in range(1, len(generated_waves)):
                    prev_wave = final_wave
                    next_wave = generated_waves[i]

                    # Calculate cross-fade samples, ensuring it does not exceed wave lengths
                    cross_fade_samples = int(cross_fade_duration * target_sample_rate)
                    cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

                    if cross_fade_samples <= 0:
                        # No overlap possible, concatenate
                        final_wave = np.concatenate([prev_wave, next_wave])
                        continue

                    # Overlapping parts
                    prev_overlap = prev_wave[-cross_fade_samples:]
                    next_overlap = next_wave[:cross_fade_samples]

                    # Fade out and fade in
                    fade_out = np.linspace(1, 0, cross_fade_samples)
                    fade_in = np.linspace(0, 1, cross_fade_samples)

                    # Cross-faded overlap
                    cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

                    # Combine
                    new_wave = np.concatenate(
                        [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
                    )

                    final_wave = new_wave

            # Create a combined spectrogram
            combined_spectrogram = np.concatenate(spectrograms, axis=1)

            yield final_wave, target_sample_rate, combined_spectrogram

        else:
            yield None, target_sample_rate, None
