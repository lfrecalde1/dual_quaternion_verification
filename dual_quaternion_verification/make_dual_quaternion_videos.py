#!/usr/bin/env python3
import argparse
import math
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def axis_angle_to_rot(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=float,
    )


def setup_ax(ax, lim=3.5, background="white"):
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_box_aspect([1, 1, 1])
    if background == "transparent":
        ax.patch.set_alpha(0.0)
    else:
        ax.set_facecolor("white")
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            axis.line.set_color((1.0, 1.0, 1.0, 0.0))
            axis._axinfo["grid"]["color"] = (1.0, 1.0, 1.0, 0.0)
            axis._axinfo["tick"]["inward_factor"] = 0.0
            axis._axinfo["tick"]["outward_factor"] = 0.0
    ax.view_init(elev=22, azim=-50)


def draw_frame(ax, origin, rot, scale=1.0):
    basis = np.eye(3)
    colors = ["#e63946", "#2a9d8f", "#457b9d"]
    labels = ["x_b", "y_b", "z_b"]
    for i in range(3):
        vec = rot @ basis[:, i] * scale
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            vec[0],
            vec[1],
            vec[2],
            color=colors[i],
            linewidth=2.0,
            arrow_length_ratio=0.15,
        )
        ax.text(origin[0] + vec[0], origin[1] + vec[1], origin[2] + vec[2], labels[i], color=colors[i], fontsize=10)


def render_orientation_frames(frames_dir, n_frames, background):
    axis = np.array([1.0, 0.0, 0.0])
    thetas = np.linspace(math.pi, 0.0, n_frames)
    for k, th in enumerate(thetas):
        fig = plt.figure(figsize=(6, 6))
        if background == "transparent":
            fig.patch.set_alpha(0.0)
        else:
            fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111, projection="3d")
        setup_ax(ax, lim=1.8, background=background)
        ax.set_title("Orientation: pi -> 0", pad=16)

        draw_frame(ax, np.zeros(3), np.eye(3), scale=1.0)
        rot = axis_angle_to_rot(axis, th)
        draw_frame(ax, np.zeros(3), rot, scale=1.3)

        out = frames_dir / f"frame_{k:04d}.png"
        fig.savefig(out, dpi=160, transparent=(background == "transparent"), facecolor=fig.get_facecolor())
        plt.close(fig)


def render_translation_frames(frames_dir, n_frames, background):
    axis = np.array([1.0, 0.0, 0.0])
    s = np.linspace(0.0, 1.0, n_frames)
    thetas = np.linspace(math.pi, 0.0, n_frames)
    for k, (u, th) in enumerate(zip(s, thetas)):
        fig = plt.figure(figsize=(6, 6))
        if background == "transparent":
            fig.patch.set_alpha(0.0)
        else:
            fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111, projection="3d")
        setup_ax(ax, lim=3.5, background=background)
        ax.set_title("Translation: (3,3,3) -> (0,0,0), orientation pi -> 0", pad=16)

        start = np.array([3.0, 3.0, 3.0])
        pos = (1.0 - u) * start
        path = np.vstack([start, pos, np.zeros(3)])
        ax.plot(path[:, 0], path[:, 1], path[:, 2], "--", color="gray", alpha=0.5, linewidth=1.2)
        ax.scatter([start[0], 0.0, pos[0]], [start[1], 0.0, pos[1]], [start[2], 0.0, pos[2]], c=["black", "black", "orange"], s=[20, 20, 36])

        draw_frame(ax, np.zeros(3), np.eye(3), scale=0.9)
        rot = axis_angle_to_rot(axis, th)
        draw_frame(ax, pos, rot, scale=0.9)

        out = frames_dir / f"frame_{k:04d}.png"
        fig.savefig(out, dpi=160, transparent=(background == "transparent"), facecolor=fig.get_facecolor())
        plt.close(fig)


def encode_alpha_mov(frames_dir, output_path, fps):
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-c:v",
        "qtrle",
        "-pix_fmt",
        "argb",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Generate transparent videos for dual quaternion orientation/translation motion.")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--background", choices=["white", "transparent"], default="white")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ori_frames = out_dir / "_frames_orientation"
    trn_frames = out_dir / "_frames_translation"
    ori_frames.mkdir(parents=True, exist_ok=True)
    trn_frames.mkdir(parents=True, exist_ok=True)

    render_orientation_frames(ori_frames, args.frames, args.background)
    render_translation_frames(trn_frames, args.frames, args.background)

    orientation_mov = out_dir / "orientation_pi_to_zero_transparent.mov"
    translation_mov = out_dir / "translation_3to0_with_pi_orientation_transparent.mov"
    encode_alpha_mov(ori_frames, orientation_mov, args.fps)
    encode_alpha_mov(trn_frames, translation_mov, args.fps)

    if not args.keep_frames:
        shutil.rmtree(ori_frames, ignore_errors=True)
        shutil.rmtree(trn_frames, ignore_errors=True)

    print(f"Saved: {orientation_mov}")
    print(f"Saved: {translation_mov}")


if __name__ == "__main__":
    main()
