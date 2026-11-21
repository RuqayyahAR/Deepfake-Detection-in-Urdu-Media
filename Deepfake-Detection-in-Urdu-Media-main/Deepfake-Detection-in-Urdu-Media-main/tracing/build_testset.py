"""Reproducible test set for source tracing. Synthetic only: no real people,
no deepfakes. Audio is a per-reference sequence of tones (stand-in for speech,
which we do not have yet)."""
import subprocess
import os


def ff(args):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y"] + args, check=True)


os.makedirs("data/refs", exist_ok=True)
os.makedirs("data/suspects", exist_ok=True)

VIDEO = ["testsrc2", "mandelbrot", "cellauto", "life", "gradients",
         "smptebars", "rgbtestsrc", "pal75bars"]


def tone_seq(seed):
    x = f"(floor(t*3)*(0.0527+{seed}*0.0131))"
    return f"0.3*sin(2*PI*(220+700*({x}-floor({x})))*t)"


for i, v in enumerate(VIDEO):
    ff(["-f", "lavfi", "-i", f"{v}=s=320x240:r=25",
        "-f", "lavfi", "-i", f"aevalsrc={tone_seq(i+1)}:s=44100:d=20",
        "-map", "0:v", "-map", "1:a", "-t", "20", "-c:v", "libx264",
        "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
        f"data/refs/ref{i}.mp4"])

# Unseen control: different content
ff(["-f", "lavfi", "-i", "cellauto=s=320x240:r=25:seed=99",
    "-f", "lavfi", "-i", f"aevalsrc={tone_seq(99)}:s=44100:d=20",
    "-map", "0:v", "-map", "1:a", "-t", "20", "-c:v", "libx264",
    "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac",
    "data/unseen_control.mp4"])

R, S = "data/refs/ref0.mp4", "data/suspects"

# Recompressed (should MATCH)
ff(["-i", R, "-c:v", "libx264", "-crf", "23", "-c:a", "aac",
    "-b:a", "64k", f"{S}/recomp_crf23.mp4"])

# Recompressed + downscaled (should mostly MATCH)
ff(["-i", R, "-vf", "scale=240:180", "-c:v", "libx264", "-crf", "32",
    "-c:a", "aac", "-b:a", "48k", f"{S}/recomp_scale240_crf32.mp4"])

# Recompressed chain via mpeg4 (should mostly MATCH)
ff(["-i", R, "-vf", "scale=160:120", "-c:v", "mpeg4", "-q:v", "12",
    "-c:a", "aac", "-b:a", "32k", f"{S}/recomp_chain_messaging.mp4"])

# Trimmed 5-8s (should MATCH for kept seconds)
ff(["-ss", "5", "-t", "8", "-i", R, "-c:v", "libx264", "-crf", "20",
    "-c:a", "aac", f"{S}/trim_5s_8s.mp4"])

# Splice: ref0 + ref5 segment + ref0 (middle 3s is VIDEO_ALTERED)
ff(["-i", R, "-i", "data/refs/ref5.mp4", "-filter_complex",
    "[0:v]trim=0:10,setpts=PTS-STARTPTS[a];[1:v]trim=0:3,setpts=PTS-STARTPTS,"
    "scale=320:240[b];[0:v]trim=13:20,setpts=PTS-STARTPTS[c];"
    "[a][b][c]concat=n=3:v=1:a=0[v]",
    "-map", "[v]", "-map", "0:a", "-c:v", "libx264", "-crf", "20",
    "-c:a", "aac", f"{S}/splice_video_10_13.mp4"])

# Region edit: red box drawn for 6-12s (VIDEO_ALTERED in that range)
ff(["-i", R,
    "-vf", "drawbox=x=40:y=40:w=140:h=100:color=red:t=fill:"
           "enable='between(t,6,12)'",
    "-c:v", "libx264", "-crf", "20", "-c:a", "aac",
    f"{S}/region_edit_6_12.mp4"])

# Audio swapped: ref0 video + ref3 audio (AUDIO_ALTERED throughout)
ff(["-i", R, "-i", "data/refs/ref3.mp4", "-map", "0:v", "-map", "1:a",
    "-c:v", "copy", "-c:a", "aac", f"{S}/audio_swapped.mp4"])

print("test set built")
