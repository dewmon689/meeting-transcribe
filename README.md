# Meeting Transcribe: local transcription with speaker labels

Turns a meeting recording into a timestamped transcript with speaker labels. It runs entirely on
your laptop: there's no account, no length cap, and no audio or text leaves the machine. The only
network use is a one-time model download.

- **Transcription:** Whisper (OpenAI's open-source speech model, run with `faster-whisper`)
- **Who-spoke-when:** voiceprints (SpeechBrain ECAPA) grouped into speakers
- **Input:** mp3, m4a, wav, mp4, … (anything with an audio track, including Teams/Zoom recordings and phone voice memos)
- **Output:** `MEETING.transcript.txt` (speaker-labelled paragraphs), `MEETING.srt` (subtitles), `MEETING.json`

---

## 1. One-time setup (~15 minutes)

You need **Python 3.12** and about **4 GB of disk**: ~2 GB of Python packages plus a 1.6 GB model.
Download it from GitHub (**Code → Download ZIP**) and unzip it into your home directory. GitHub names
the folder `meeting-transcribe-main`, so rename it to `meeting-transcribe`. The commands below assume that name.

> On a work laptop, check that IT allows installing Python packages. If huggingface.co is blocked
> (the model download fails), see [Troubleshooting](#troubleshooting).

### Windows (PowerShell)

Install Python 3.12 from python.org (tick **"Add python.exe to PATH"**), from Company Portal, or with
`winget install Python.Python.3.12`. Then:

```powershell
cd $HOME\meeting-transcribe
py -3.12 -m venv $HOME\transcribe-venv
& $HOME\transcribe-venv\Scripts\python -m pip install --upgrade pip

# Most laptops (no NVIDIA GPU):
& $HOME\transcribe-venv\Scripts\pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
# NVIDIA GPU instead: same line with  .../whl/cu128  in place of  .../whl/cpu

& $HOME\transcribe-venv\Scripts\pip install -r requirements.txt
```

Optional shortcut, so you can type `transcribe` from anywhere. Run `notepad $PROFILE` and add:

```powershell
function transcribe { & $HOME\transcribe-venv\Scripts\python $HOME\meeting-transcribe\transcribe.py @args }
```

If PowerShell refuses to load the profile because of a corporate execution policy, skip the shortcut
and use the full command: `& $HOME\transcribe-venv\Scripts\python $HOME\meeting-transcribe\transcribe.py ...`

### macOS (Terminal)

Install Python 3.12 from python.org or with `brew install python@3.12`. Then:

```bash
cd ~/meeting-transcribe
python3.12 -m venv ~/transcribe-venv
~/transcribe-venv/bin/pip install --upgrade pip
~/transcribe-venv/bin/pip install torch==2.11.0 torchaudio==2.11.0
~/transcribe-venv/bin/pip install -r requirements.txt
echo 'alias transcribe="~/transcribe-venv/bin/python ~/meeting-transcribe/transcribe.py"' >> ~/.zshrc
```

Macs run on the CPU, because the Whisper engine has no Apple-GPU support. Apple Silicon is fast
enough. Intel Macs will be slow.

### Linux

Same as macOS, but install torch with `--index-url https://download.pytorch.org/whl/cpu` (or `/whl/cu128` for NVIDIA).

---

## 2. Using it

```
transcribe MEETING.mp3 --speakers 5 --prompt "Names: Priya, Marcus, Jen. Terms: SOC 2, Kubernetes"
```

**Recommended workflow:**

1. **Find where the meeting ends.** Recordings often keep running after the meeting is over (it is
   easy to capture an hour of hallway or car conversation). Pass `--end 1:56:15` to cut that off.
2. **Run it** with `--speakers N` (the number of people who actually talk) and a `--prompt` listing
   names and jargon, which fixes most spelling errors. The first run downloads the model.
3. **Work out who's who.** The output says `Speaker 1`, `Speaker 2`, … Skim it for clues: people
   address each other by name, take roll, introduce themselves.
   - Shortcut: paste the transcript into your **company-approved** AI assistant and ask: *"Identify
     each speaker by name from context. Give me a mapping like 1=Name, with the line that proves each."*
     Pasting it into a personal AI account defeats the point of transcribing locally.
4. **Apply the names** by rerunning with the same arguments plus
   `--names "1=Priya,2=Marcus,4=Jen"`. This takes seconds because both slow steps are cached.
   Speakers you don't list stay `Speaker N`.

**Options:**

| Flag | What it does |
|---|---|
| `--speakers N` | Number of people speaking. **Use it.** Without it the tool guesses, which works on long meetings but often misses on short clips |
| `--names "1=A,2=B"` | Replace `Speaker N` with real names |
| `--prompt "..."` | Names, places and jargon so Whisper spells them right |
| `--start` / `--end` | Only transcribe part of the file (`h:mm:ss` or seconds) |
| `--model large-v3` | Most accurate model. It's the GPU default. On CPU the default is `large-v3-turbo`, which is much faster and nearly as good |
| `--device cpu` | Force CPU, e.g. if GPU setup gives DLL/CUDA errors |
| `--no-speakers` | Plain transcript, no speaker labels |
| `--out DIR` | Put the output somewhere other than next to the recording |

**Clean up afterwards:** the run also leaves `MEETING.whisper.json` and `MEETING.speakers.json`
caches next to the outputs. They contain the full transcript text, so delete them along with
everything else when you're done with a confidential meeting.

---

## Speed and accuracy (measured 2026-11-05)

| Machine | Model | Speed |
|---|---|---|
| Laptop CPU, 8-core Intel i9-9880H | large-v3-turbo | **1.9× real time**: a 1-hour meeting takes about 30 min. Peak RAM ~3.3 GB |
| NVIDIA Quadro RTX 3000 (6 GB) | large-v3 | ~9× real time: a 1-hour meeting takes about 7 min |

- **Text quality** was on par with a commercial cloud service on a real 2½-hour meeting. The CPU turbo model occasionally
  drops punctuation or capitals for a sentence.
- **Speaker labels** closely matched that service's labels (adjusted Rand index 0.84; 1.0 would be
  identical). Expect worse results on clips under ~10 minutes, and always pass `--speakers`.
- **Known limitation:** a quick interjection ("Yes." / "Okay.") can get absorbed into the previous
  speaker's paragraph.

**What's been tested:** Linux (CPU and NVIDIA). It was written to work on Windows and macOS (UTF-8
output, no symlinks, no ffmpeg needed) but hasn't been run on either yet.

---

## Troubleshooting

- **`Warning: You are sending unauthenticated requests to the HF Hub`**: harmless.
- **Model download blocked by a proxy:** run it once on any machine that can reach huggingface.co.
  Copy that machine's `~/.cache/huggingface` and `~/.cache/speechbrain` folders into your home folder
  (on Windows, `C:\Users\YOU\.cache\...`). Then set `HF_HUB_OFFLINE=1` before running
  (PowerShell: `$env:HF_HUB_OFFLINE=1`). Offline mode is tested and works.
- **`open() got an unexpected keyword argument 'metadata_errors'`**: PyAV got upgraded past 18.x.
  Reinstall it with `pip install av==18.1.0`.
- **Windows + NVIDIA: `cublas64_12.dll` / `cudnn` not found**: use `--device cpu`, or reinstall torch
  from the `cu128` index. The script imports torch first so that its bundled CUDA libraries are found.
- **Out of memory on a GPU with less than 6 GB:** use `--device cpu`.
