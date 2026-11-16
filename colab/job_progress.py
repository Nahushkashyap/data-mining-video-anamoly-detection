"""Progress of the Colab extraction job. Safe to run any time.

While the job is running the kernel is busy (a second `colab exec` would just queue
behind it), so check from a separate process over ssh:
    colab ssh -s vad -- python3 /content/job_progress.py
When no job is running, this also works:
    colab exec -s vad -f colab/job_progress.py
"""
import collections
import csv
import glob
import os

DRIVE_ROOT = os.environ.get("VAD_DRIVE_ROOT", "/content/drive/MyDrive/vad")
man = f"{DRIVE_ROOT}/manifests/ucf_subset.csv"
if not os.path.exists(man):
    print("no manifest yet: the job has not started (or Drive is not mounted)")
else:
    rows = list(csv.DictReader(open(man)))
    vid_ok, feat_ok, need = collections.Counter(), collections.Counter(), collections.Counter()
    for r in rows:
        need[r["lane"]] += 1
        p = os.path.join(DRIVE_ROOT, "ucf_videos", r["rel_path"])
        if os.path.exists(p) and os.path.getsize(p) == int(r["bytes"]):
            vid_ok[r["lane"]] += 1
        if os.path.exists(f"{DRIVE_ROOT}/features/x3d_s/{r['video_id']}.npy"):
            feat_ok[r["lane"]] += 1
    for lane in sorted(need):
        print(f"{lane:4s} videos {vid_ok[lane]:4d}/{need[lane]}   features {feat_ok[lane]:4d}/{need[lane]}")
    print("in-progress checkpoints:", len(glob.glob(f"{DRIVE_ROOT}/features/x3d_s/_partial/*.npy")))
for name in ("ucf_videos/download_log.txt", "run_log.txt"):
    p = os.path.join(DRIVE_ROOT, name)
    if os.path.exists(p):
        print(f"\n--- last lines of {name}")
        print("".join(open(p).readlines()[-4:]), end="")
