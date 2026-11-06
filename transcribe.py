#!/usr/bin/env python3
"""Local meeting transcription: Whisper + speaker labels. Runs on Windows, macOS, or Linux.

Nothing leaves the machine and there is no length cap. Uses an NVIDIA GPU if PyTorch can see one,
otherwise the CPU.

  python transcribe.py MEETING.mp3                       # auto-detect number of speakers
  python transcribe.py MEETING.mp3 --speakers 5          # known head-count (more accurate)
  python transcribe.py MEETING.mp3 --end 1:30:00 --prompt "Names: Priya, Marcus, Jen"
  python transcribe.py MEETING.mp3 --speakers 5 --names "1=Priya,3=Marcus"   # rename; reruns instantly

Writes MEETING.transcript.txt (speaker-labelled paragraphs), .srt and .json next to the input (or --out DIR).
"""
import argparse, gc, json, os, sys, time

import numpy as np

SR = 16000


def hms(t):
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}"


def parse_time(s):
    if s is None:
        return None
    secs = 0.0
    for p in str(s).split(":"):
        secs = secs * 60 + float(p)
    return secs


def pick_device(want):
    # PyTorch is the single source of truth: a CUDA build of torch ships the cuBLAS/cuDNN libraries
    # Whisper needs, and importing it first makes them loadable. CPU-only torch means CPU for both.
    import torch
    has_gpu = torch.cuda.is_available()
    if want == "cuda" and not has_gpu:
        sys.exit("--device cuda requested but PyTorch sees no CUDA GPU (CPU-only torch build?)")
    return "cuda" if want in ("auto", "cuda") and has_gpu else "cpu"


def decode(path, start, end):
    """Decode any audio/video file to 16 kHz mono float32 (PyAV, no ffmpeg install needed)."""
    from faster_whisper.audio import decode_audio
    audio = decode_audio(path, sampling_rate=SR)
    a = int((start or 0) * SR)
    b = int(end * SR) if end else len(audio)
    return np.ascontiguousarray(audio[a:b], dtype=np.float32)


def load_cache(path, key):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            c = json.load(f)
        if c.get("key") == key:
            return c["data"]
    return None


def save_cache(path, key, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"key": key, "data": data}, f)


def transcribe(audio, model_name, prompt, language, device):
    from faster_whisper import WhisperModel
    if device == "cuda":
        model = WhisperModel(model_name, device="cuda", compute_type="int8_float16")
    else:
        model = WhisperModel(model_name, device="cpu", compute_type="int8",
                             cpu_threads=os.cpu_count() or 4)
    segs, _ = model.transcribe(audio, language=language, vad_filter=True, beam_size=5,
                               initial_prompt=prompt, condition_on_previous_text=False)
    out = []
    for s in segs:
        text = s.text.strip()
        # drop Whisper's classic hallucinated repeats ("Right. Right. Right.")
        if out and text == out[-1]["text"] and len(text) < 20:
            continue
        out.append({"start": s.start, "end": s.end, "text": text})
        print(f"\r  transcribed to {hms(s.end)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    del model
    gc.collect()
    return out


def estimate_speakers(S, max_speakers):
    """Eigengap heuristic on a row-pruned affinity matrix (keep each segment's closest neighbours)."""
    from scipy.sparse.csgraph import laplacian
    p = max(10, len(S) // 10)
    A = np.zeros_like(S)
    for i, row in enumerate(S):
        keep = np.argsort(row)[-p:]
        A[i, keep] = row[keep]
    A = (A + A.T) / 2
    ev = np.sort(np.linalg.eigvalsh(laplacian(A, normed=True)))[:max_speakers + 1]
    gaps = np.diff(ev)
    return int(np.argmax(gaps[1:]) + 2)    # at least 2 speakers


def diarize(audio, segs, n_speakers, max_speakers, device):
    """Return a speaker number (1, 2, ... in order of first appearance) for every segment."""
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
    from sklearn.cluster import SpectralClustering

    device = "cuda:0" if device == "cuda" else device   # speechbrain wants an explicit index
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(os.path.expanduser("~"), ".cache", "speechbrain", "spkrec-ecapa-voxceleb"),
        run_opts={"device": device},
        local_strategy=LocalStrategy.COPY)      # symlinks need admin rights on Windows
    max_len = 10 * SR           # long segments blow up memory; 10 s is plenty for a voiceprint
    embs, idx = [], []
    for i, s in enumerate(segs):
        a, b = int(s["start"] * SR), min(int(s["end"] * SR), len(audio))  # Whisper can overshoot the end
        if b - a < SR:          # <1 s is too short for a reliable voiceprint
            continue
        if b - a > max_len:     # use the middle of the segment
            a += (b - a - max_len) // 2
            b = a + max_len
        wav = torch.from_numpy(audio[a:b]).unsqueeze(0)
        with torch.no_grad():
            e = enc.encode_batch(wav.to(device)).squeeze().cpu().numpy()
        embs.append(e / np.linalg.norm(e))
        idx.append(i)
        print(f"\r  voiceprints {len(embs)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    if len(embs) < 2:
        return [1] * len(segs)
    X = np.stack(embs)
    # Spectral clustering on cosine similarity. Scored against a commercial service's labels on a real meeting
    # it beat agglomerative (ARI 0.84 vs 0.63); average linkage collapsed long meetings into one speaker.
    S = np.clip(X @ X.T, 0, 1)
    if not n_speakers:
        n_speakers = estimate_speakers(S, max_speakers)
        print(f"  estimated {n_speakers} speakers (pass --speakers N if you know the head-count)",
              file=sys.stderr)
    found = SpectralClustering(n_clusters=min(n_speakers, len(X)), affinity="precomputed",
                               random_state=0).fit_predict(S)
    order, labels = {}, [None] * len(segs)
    for i, l in zip(idx, found):
        labels[i] = order.setdefault(int(l), len(order) + 1)
    # short segments inherit the nearest labelled neighbour in time
    labelled = [i for i in range(len(segs)) if labels[i] is not None]
    for i, s in enumerate(segs):
        if labels[i] is None:
            j = min(labelled, key=lambda k: abs(segs[k]["start"] - s["start"]))
            labels[i] = labels[j]
    return labels


def parse_names(spec):
    names = {}
    for part in (spec or "").split(","):
        if part.strip():
            n, _, name = part.partition("=")
            names[int(n.strip())] = name.strip()
    return names


def write_outputs(segs, base, offset):
    for s in segs:
        s["start"] += offset
        s["end"] += offset
    # merge consecutive segments from the same speaker into paragraphs
    paras = []
    for s in segs:
        if paras and paras[-1]["speaker"] == s["speaker"] and s["start"] - paras[-1]["end"] < 3:
            paras[-1]["text"] += " " + s["text"]
            paras[-1]["end"] = s["end"]
        else:
            paras.append(dict(s))
    with open(base + ".transcript.txt", "w", encoding="utf-8") as f:
        for p in paras:
            f.write(f"{p['speaker']}  {hms(p['start'])}\n{p['text']}\n\n")

    def srt_t(t):
        ms = min(999, int(round((t - int(t)) * 1000)))
        return f"{hms(t).zfill(8)},{ms:03d}"
    with open(base + ".srt", "w", encoding="utf-8") as f:
        for n, s in enumerate(segs, 1):
            f.write(f"{n}\n{srt_t(s['start'])} --> {srt_t(s['end'])}\n[{s['speaker']}] {s['text']}\n\n")
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(segs, f, indent=1, ensure_ascii=False)
    return len(paras), len({s["speaker"] for s in segs})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", help="mp3, m4a, wav, mp4, ... (anything with an audio track)")
    ap.add_argument("--speakers", type=int, help="known number of speakers (recommended)")
    ap.add_argument("--max-speakers", type=int, default=15, help="upper bound when estimating the speaker count")
    ap.add_argument("--names", help='rename speakers, e.g. "1=Priya,3=Marcus" (unlisted ones stay "Speaker N")')
    ap.add_argument("--model", help="Whisper model (default: large-v3 on GPU, large-v3-turbo on CPU)")
    ap.add_argument("--language", default="en")
    ap.add_argument("--prompt", help="vocabulary hint: names, places, jargon")
    ap.add_argument("--start", help="start time (s or h:mm:ss)")
    ap.add_argument("--end", help="end time (s or h:mm:ss) - cut off anything recorded after the meeting")
    ap.add_argument("--no-speakers", action="store_true", help="skip speaker labelling")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--out", help="output directory (default: next to input)")
    a = ap.parse_args()

    device = pick_device(a.device)
    model = a.model or ("large-v3" if device == "cuda" else "large-v3-turbo")
    start, end = parse_time(a.start), parse_time(a.end)
    stem = os.path.splitext(os.path.basename(a.audio))[0]
    base = os.path.join(a.out or os.path.dirname(os.path.abspath(a.audio)), stem)
    names = parse_names(a.names)

    t0 = time.time()
    print(f"decoding {a.audio} (device: {device}, model: {model})", file=sys.stderr)
    audio = decode(a.audio, start, end)
    dur = len(audio) / SR

    # Cache the slow passes so renaming speakers, or a crash in the speaker step, costs nothing.
    key = {"start": start, "end": end, "model": model, "prompt": a.prompt}
    segs = load_cache(base + ".whisper.json", key)
    if segs is not None:
        print(f"  {hms(dur)} of audio; reusing earlier transcription", file=sys.stderr)
    else:
        print(f"  {hms(dur)} of audio; transcribing (first run downloads the model)", file=sys.stderr)
        segs = transcribe(audio, model, a.prompt, a.language, device)
        save_cache(base + ".whisper.json", key, segs)

    if a.no_speakers:
        labels = [1] * len(segs)
    else:
        skey = dict(key, speakers=a.speakers, max_speakers=a.max_speakers)
        labels = load_cache(base + ".speakers.json", skey)
        if labels is not None:
            print("  reusing earlier speaker labels", file=sys.stderr)
        else:
            print("  labelling speakers", file=sys.stderr)
            labels = diarize(audio, segs, a.speakers, a.max_speakers, device)
            save_cache(base + ".speakers.json", skey, labels)
    for s, n in zip(segs, labels):
        s["speaker"] = names.get(n, f"Speaker {n}")

    n_para, n_spk = write_outputs(segs, base, start or 0)
    el = time.time() - t0
    print(f"done in {hms(el)} ({dur / max(el, 1e-9):.1f}x real time): {n_para} paragraphs, {n_spk} speakers",
          file=sys.stderr)
    print(base + ".transcript.txt")


if __name__ == "__main__":
    main()
