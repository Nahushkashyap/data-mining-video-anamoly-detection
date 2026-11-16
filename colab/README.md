# Running the UCF-Crime extraction with the Colab CLI

Everything runs on a Colab VM. Videos, features and logs go to `MyDrive/vad/`.
All commands are run **from the project folder on the Mac**.

## One-time setup
The CLI (v0.8.0) is installed in the project venv: `source venv/bin/activate` puts `colab` on the PATH
(or call `venv/bin/colab`). The first `colab` command opens a Google sign-in (OAuth) flow.

## Start a session (T4 is enough; the job is limited by download speed, not GPU)
```bash
colab new -s vad --gpu T4
colab drivemount -s vad               # approve Drive access in the browser (same Google account)
colab upload -s vad scripts/ucf_download.py /content/ucf_download.py
colab upload -s vad scripts/extract_x3d.py  /content/extract_x3d.py
colab upload -s vad colab/job_progress.py   /content/job_progress.py
```
Note: put your own Hugging Face token in `HF_TOKEN` inside `scripts/ucf_download.py` (or the `HF_TOKEN` env var) before uploading; it is optional and only avoids rate limits. Never commit it.

## Run the job (download + extract, lanes set at the top of colab/job_extract.py)
```bash
caffeinate -i colab exec -s vad --timeout 0 -f colab/job_extract.py
```
- Runs in the foreground on purpose: a Colab VM stays alive only while its kernel is busy.
- Output streams to your terminal and to `MyDrive/vad/run_log.txt`.
- If anything dies (VM reclaimed, Wi-Fi drop, Ctrl-C): rerun the same command. If the session is gone,
  repeat the "Start a session" block first. Finished videos/features are skipped, the video in progress
  resumes from its last 2-minute checkpoint, and only one downloader ever runs per folder.

## Check progress (second terminal, while the job runs)
```bash
colab ssh -s vad -i <your-ssh-key> -- python3 /content/job_progress.py
colab usage                            # compute-unit burn rate and balance
```
Without an SSH key: the job's own output (terminal, or `MyDrive/vad/run_log.txt` in Drive) shows `[n/1000]` per video.
`colab exec` cannot be used for this while the job runs: the kernel runs one request at a time, so it would queue.

Speed seen on 19 Nov (T4, 2 vCPU): ~380 ms/clip while the downloader was busy (CPU-bound decode + 12 download
threads), ~65-90 ms/clip once downloads were done. The GPU is not the bottleneck.

## Finish
When the job prints `all videos in the selected lanes have features.`:
```bash
colab ssh -s vad -i <your-ssh-key> -- python3 /content/job_progress.py   # expect L1 600/600, L2 400/400
colab stop -s vad                      # releases the VM so it stops using compute units
```
For the optional extra 400 normals, set `LANES = "L2X"` in `colab/job_extract.py` and run the job again.

---

# Home-security audio head (22 Nov 2026): `colab/job_audio_home.py`

Trains the audio Stage 1 head for **scream, glass break, loud bang, gunshot, explosion, alarm** on FSD50K
(frozen EfficientAT mn10 embeddings of 1 s windows + a small MIL head). No Drive needed: everything stays on
the VM and the results are copied back with `colab download`.

```bash
colab new -s vadaudio --gpu T4
COPYFILE_DISABLE=1 tar --exclude=__pycache__ -czf /tmp/vad_home_bundle.tar.gz vad models_cache/efficientat scripts/ucf_download.py
colab upload -s vadaudio /tmp/vad_home_bundle.tar.gz /content/vad_home_bundle.tar.gz
echo '{"tag": "full", "limit": null, "epochs": 40, "clap": true}' > /tmp/home_cfg.json   # "limit": 80 for a smoke test
colab upload -s vadaudio /tmp/home_cfg.json /content/home_cfg.json
caffeinate -i colab exec -s vadaudio --timeout 0 -f colab/job_audio_home.py
colab download -s vadaudio /content/home_out/full/home_head_v1.pt models_cache/efficientat/home_head_v1.pt
for f in home_head_eval.json two_stage_eval.json log.txt; do colab download -s vadaudio /content/home_out/full/$f data/audio_eval/results/home_$f; done
colab stop -s vadaudio
```
- Selects ~11k clips from the FSD50K labels (HF mirror `Fhrozen/FSD50k`): all target-class positives plus
  household hard negatives (speech, music, TV-like sounds, doors, knocks, footsteps, dogs, fireworks, clinking glass,
  drums...). Downloads only those clips (~25 clips/s with the HF token from `scripts/ucf_download.py`).
- Train/val = FSD50K dev split, test = FSD50K eval split (different uploaders). Thresholds come from val only.
- Measured run (22 Nov, T4): download 11,063 clips (6.7 GB) in ~10 min, extraction of 146,831 windows ~9 min, 3-seed training ~4 min, CLAP two-stage evaluation ~1 min; about 0.85 compute units in total. Results: eval mAP 0.661 ± 0.003 (zero-shot 0.409); see reports/Audio visual video anomaly detection.md (Addendum).
- The first full run was OOM-killed during extraction (decode prefetch held every clip in RAM); `extract` now decodes at most 64 clips ahead (peak RAM ~2.3 GB).
- In the app: the head is used when `models_cache/efficientat/home_head_v1.pt` exists. `VAD_AUDIO_MODEL=zeroshot ./run_webui.sh`
  restores the original zero-shot trigger and CLAP prompts. A full code snapshot from before the change is in
  `backups/pre_home_audio_2026-11-22.tar.gz`.
