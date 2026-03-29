#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np


EPS = 1e-12
plt.rc("text", usetex=False)


def normalize_quaternion(q):
    n = np.linalg.norm(q)
    if n < EPS:
        raise ValueError("Quaternion norm is near zero.")
    return q / n


def quaternion_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quaternion_error(qd, q):
    qd = normalize_quaternion(qd)
    q = normalize_quaternion(q)
    qe = quaternion_multiply(qd, quaternion_conjugate(q))  # q_e = q_d ⊗ q^{-1}
    qe = normalize_quaternion(qe)
    if qe[0] < 0.0:
        qe = -qe
    return qe


def quaternion_distance_norm(qd, q):
    qd = normalize_quaternion(qd)
    q = normalize_quaternion(q)
    # Sign-invariant quaternion distance using only vector norms.
    return min(np.linalg.norm(qd - q), np.linalg.norm(qd + q))


def dual_quaternion_from_q_t(q, t):
    q = normalize_quaternion(q)
    t_quat = np.array([0.0, t[0], t[1], t[2]], dtype=float)
    d = 0.5 * quaternion_multiply(t_quat, q)
    return np.concatenate([q, d])


def dual_quaternion_conjugate(qd):
    return np.array(
        [qd[0], -qd[1], -qd[2], -qd[3], qd[4], -qd[5], -qd[6], -qd[7]],
        dtype=float,
    )


def dual_quaternion_multiply(a, b):
    ar, ad = a[0:4], a[4:8]
    br, bd = b[0:4], b[4:8]
    real = quaternion_multiply(ar, br)
    dual = quaternion_multiply(ar, bd) + quaternion_multiply(ad, br)
    return np.concatenate([real, dual])


def dual_quaternion_error(qd_des, qd_cur):
    qd_des_c = dual_quaternion_conjugate(qd_des)
    qe = dual_quaternion_multiply(qd_des_c, qd_cur)
    qe[0:4] = normalize_quaternion(qe[0:4])
    if qe[0] < 0.0:
        qe = -qe
    return qe


def ln_dual_orientation_vector(qe_dual):
    q_error_real = normalize_quaternion(qe_dual[0:4])
    v = q_error_real[1:4]
    vnorm = np.linalg.norm(v)
    if vnorm < EPS:
        return np.zeros(3)
    angle = 2.0 * math.atan2(vnorm, q_error_real[0])
    # Orientation part from ln(Qe): imaginary part of ln(quaternion)
    return 0.5 * angle * (v / vnorm)


def reduced_yaw_error_vectors(qe):
    qew, qex, qey, qez = qe
    denom = math.sqrt(qew * qew + qez * qez) + EPS

    q_tilde_red = np.array(
        [
            (qew * qex - qey * qez) / denom,
            (qew * qey + qex * qez) / denom,
            0.0,
        ],
        dtype=float,
    )
    q_tilde_yaw = np.array([0.0, 0.0, qez / denom], dtype=float)
    return q_tilde_red, q_tilde_yaw


def generate_demo_data(n):
    t = np.linspace(0.0, 12.0, n)

    # Desired orientation: yaw changes slowly.
    yaw_d = 0.45 * np.sin(0.35 * t)
    # Current orientation: yaw + roll/pitch disturbances.
    yaw = yaw_d + 0.25 * np.sin(1.1 * t)
    pitch = 0.20 * np.sin(0.6 * t + 0.4)
    roll = 0.15 * np.sin(0.8 * t - 0.2)

    qd = np.array([yaw_to_quat(z) for z in yaw_d])
    q = np.array([rpy_to_quat(r, p, y) for r, p, y in zip(roll, pitch, yaw)])
    return t, qd, q


def yaw_to_quat(yaw):
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=float)


def rpy_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    # ZYX convention
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=float,
    )


def axis_angle_to_quat(axis, angle_rad):
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < EPS:
        raise ValueError("Axis norm is near zero.")
    axis = axis / n
    half = 0.5 * angle_rad
    return np.array(
        [math.cos(half), axis[0] * math.sin(half), axis[1] * math.sin(half), axis[2] * math.sin(half)],
        dtype=float,
    )


def skew(v):
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=float,
    )


def rotation_log_vector_from_quaternion(q):
    q = normalize_quaternion(q)
    v = q[1:4]
    vnorm = np.linalg.norm(v)
    if vnorm < EPS:
        return np.zeros(3)
    theta = 2.0 * math.atan2(vnorm, q[0])
    return theta * (v / vnorm)


def left_jacobian_inverse_so3(phi):
    theta = np.linalg.norm(phi)
    I = np.eye(3)
    if theta < 1e-8:
        phi_hat = skew(phi)
        return I - 0.5 * phi_hat + (1.0 / 12.0) * (phi_hat @ phi_hat)

    phi_hat = skew(phi)
    a = (1.0 / (theta * theta)) - ((1.0 + math.cos(theta)) / (2.0 * theta * math.sin(theta)))
    return I - 0.5 * phi_hat + a * (phi_hat @ phi_hat)


def translation_from_dual_quaternion(qd):
    q_r = normalize_quaternion(qd[0:4])
    q_d = qd[4:8]
    t_quat = 2.0 * quaternion_multiply(q_d, quaternion_conjugate(q_r))
    return t_quat[1:4]


def generate_large_to_zero_data(n, initial_angle_deg):
    t = np.linspace(0.0, 1.0, n)
    qd = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=float), (n, 1))  # desired: zero orientation error

    angles = np.linspace(math.radians(initial_angle_deg), 0.0, n)
    q = np.array([axis_angle_to_quat(axis=[1.0, 1.0, 0.5], angle_rad=ang) for ang in angles])
    return t, qd, q


def generate_angle_sweep_data(n):
    # x-axis is the orientation error angle from 0 to pi [rad]
    theta = np.linspace(0.0, math.pi-0.001, n)
    qd = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=float), (n, 1))
    q = np.array([axis_angle_to_quat(axis=[1.0, 0.0, 0.0], angle_rad=th) for th in theta])
    return theta, qd, q


def generate_translation_to_origin_data(n):
    s = np.linspace(0.0, 1.0, n)
    td = np.zeros((n, 3))
    t = np.column_stack(
        [
            3.0 * s,
            3.0 * s,
            3.0 * s,
        ]
    )
    return s, td, t


def generate_translation_orientation_grid(n_t, n_o):
    s_t = np.linspace(0.0, 1.0, n_t)  # 0 -> (3,3,3), 1 -> origin
    theta = np.linspace(math.pi, 0.0, n_o)  # pi -> 0
    return s_t, theta


def load_csv(path):
    data = np.genfromtxt(path, delimiter=",", names=True)

    required = [
        "qd_w",
        "qd_x",
        "qd_y",
        "qd_z",
        "q_w",
        "q_x",
        "q_y",
        "q_z",
    ]
    for key in required:
        if key not in data.dtype.names:
            raise ValueError(f"CSV is missing required column: {key}")

    if "t" in data.dtype.names:
        t = np.asarray(data["t"], dtype=float)
    else:
        t = np.arange(data.shape[0], dtype=float)

    qd = np.column_stack([data["qd_w"], data["qd_x"], data["qd_y"], data["qd_z"]]).astype(float)
    q = np.column_stack([data["q_w"], data["q_x"], data["q_y"], data["q_z"]]).astype(float)
    return t, qd, q


def compute_metrics(t, qd_all, q_all):
    ln_cost_from_dual = []
    reduced_yaw_cost = []
    quat_distance_norm = []

    for qd, q in zip(qd_all, q_all):
        qd_dual = dual_quaternion_from_q_t(qd, np.zeros(3))
        q_dual = dual_quaternion_from_q_t(q, np.zeros(3))
        qe_dual = dual_quaternion_error(qd_dual, q_dual)
        qe_i = qe_dual[0:4]

        ln_vec = ln_dual_orientation_vector(qe_dual)
        q_red, q_yaw = reduced_yaw_error_vectors(qe_i)

        ln_cost_from_dual.append(float(np.linalg.norm(ln_vec)))
        reduced_yaw_cost.append(float(np.linalg.norm(q_red) + np.linalg.norm(q_yaw)))
        quat_distance_norm.append(float(quaternion_distance_norm(qd, q)))

    ln_cost = np.array(ln_cost_from_dual)
    reduced_yaw_cost = np.array(reduced_yaw_cost)
    quat_distance_norm = np.array(quat_distance_norm)
    return ln_cost, reduced_yaw_cost, quat_distance_norm


def compute_translation_metrics(td_all, t_all, qd_all, q_all):
    classical = []
    log_dual = []

    for td, t, qd, q in zip(td_all, t_all, qd_all, q_all):
        Td = dual_quaternion_from_q_t(qd, td)
        T = dual_quaternion_from_q_t(q, t)
        Te = dual_quaternion_error(Td, T)  # Te = Td^{-1} T

        p_e = translation_from_dual_quaternion(Te)
        phi = rotation_log_vector_from_quaternion(Te[0:4])
        J_inv = left_jacobian_inverse_so3(phi)
        rho = J_inv@p_e

        classical.append(float(np.linalg.norm(td - t)))
        log_dual.append(float(np.linalg.norm(rho)))

    return np.array(classical), np.array(log_dual)


def compute_translation_metrics_grid(s_t, theta, rot_axis):
    dual_log_norm = np.zeros((theta.shape[0], s_t.shape[0]))
    classical_norm = np.zeros((theta.shape[0], s_t.shape[0]))

    qd = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    td = np.zeros(3)
    rot_axis = np.asarray(rot_axis, dtype=float)

    for i, th in enumerate(theta):
        q = axis_angle_to_quat(rot_axis, th)
        for j, s in enumerate(s_t):
            t = np.array([3.0 * s, 3.0 * s, 3.0 * s], dtype=float)
            Td = dual_quaternion_from_q_t(qd, td)
            T = dual_quaternion_from_q_t(q, t)
            Te = dual_quaternion_error(Td, T)

            p_e = translation_from_dual_quaternion(Te)
            phi = rotation_log_vector_from_quaternion(Te[0:4])
            rho = left_jacobian_inverse_so3(phi) @ p_e

            dual_log_norm[i, j] = np.linalg.norm(rho)
            classical_norm[i, j] = np.linalg.norm(td - t)

    return dual_log_norm, classical_norm


def fancy_plots_2():
    pts_per_inch = 72.27
    text_width_in_pts = 300.0
    text_width_in_inches = text_width_in_pts / pts_per_inch
    golden_ratio = 0.618
    inverse_latex_scale = 2
    fig_proportion = 3.0 / 3.0
    csize = inverse_latex_scale * fig_proportion * text_width_in_inches
    fig_size = (1.0 * csize, (0.7 * csize) / golden_ratio * golden_ratio)
    text_size = inverse_latex_scale * 10
    tick_size = inverse_latex_scale * 8

    params = {
        "backend": "ps",
        "axes.labelsize": text_size,
        "legend.fontsize": tick_size,
        "legend.handlelength": 2.5,
        "legend.borderaxespad": 0,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "font.family": "serif",
        "font.size": text_size,
        "ps.usedistiller": "xpdf",
        "text.usetex": False,
        "figure.figsize": fig_size,
    }
    plt.rcParams.update(params)
    plt.clf()
    fig = plt.figure(1, figsize=fig_size)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.13, hspace=0.05, wspace=0.02)
    plt.ioff()
    gs = gridspec.GridSpec(2, 1, figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    return fig, ax1, ax2


def plot_results(t, ln_cost, reduced_yaw_cost, out, x_label="t"):
    fig, ax1, ax2 = fancy_plots_2()

    ax1.plot(t, ln_cost, label="Our Method", linewidth=1.8)
    ax1.plot(t, reduced_yaw_cost, label="State of the art", linewidth=1.8)
    ax1.set_ylabel("Orientation Cost")
    ax1.set_xlabel(x_label)
    ax1.grid(True, alpha=0.3)

    # Keep second axis as a clean decorative panel, matching fancy_plots_2 layout.
    ax2.axis("off")
    rect = patches.Rectangle((0.0, 0.0), 1.0, 1.0, linewidth=0.0, edgecolor="none", facecolor="white")
    ax2.add_patch(rect)

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"Saved plot: {out}")


def plot_translation_results(x, classical, log_dual, out):
    fig, ax1, ax2 = fancy_plots_2()

    ax1.plot(x, classical, label="Classical ||td - t||", linewidth=1.8)
    ax1.plot(x, log_dual, label="Dual log ||rho||", linewidth=1.8)
    ax1.set_ylabel("Translation Error Norm [m]")
    ax1.set_xlabel("Path parameter (0,0,0) -> (3,3,3)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    ax2.axis("off")
    rect = patches.Rectangle((0.0, 0.0), 1.0, 1.0, linewidth=0.0, edgecolor="none", facecolor="white")
    ax2.add_patch(rect)

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"Saved plot: {out}")


def _apply_view(ax, view):
    if view == "top":
        ax.view_init(elev=90, azim=-90)
    elif view == "side":
        ax.view_init(elev=10, azim=0)
    elif view == "front":
        ax.view_init(elev=10, azim=90)
    else:  # iso
        ax.view_init(elev=28, azim=-55)


def plot_translation_orientation_3d(s_t, theta, dual_log_norm, classical_norm, out, view="iso", style="surface"):
    S, TH = np.meshgrid(s_t, theta)
    fig = plt.figure(figsize=(14, 6.4))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    if style == "wireframe":
        surf1 = ax1.plot_wireframe(S, TH, dual_log_norm, rstride=3, cstride=3, linewidth=0.7, color="teal")
    else:
        surf1 = ax1.plot_surface(S, TH, dual_log_norm, cmap="viridis", edgecolor="none", alpha=0.95)
    ax1.set_title("Dual-Quaternion Log Translation Norm", fontsize=14, pad=10)
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.tick_params(axis="z", which="major", labelsize=16)
    if style == "surface":
        cb1 = fig.colorbar(surf1, ax=ax1, shrink=0.82, pad=0.02)
        cb1.ax.tick_params(labelsize=16)
    _apply_view(ax1, view)

    if style == "wireframe":
        surf2 = ax2.plot_wireframe(S, TH, classical_norm, rstride=3, cstride=3, linewidth=0.7, color="purple")
    else:
        surf2 = ax2.plot_surface(S, TH, classical_norm, cmap="plasma", edgecolor="none", alpha=0.95)
    ax2.set_title("Classical Translation", fontsize=14, pad=10)
    ax2.tick_params(axis="both", which="major", labelsize=16)
    ax2.tick_params(axis="z", which="major", labelsize=16)
    if style == "surface":
        cb2 = fig.colorbar(surf2, ax=ax2, shrink=0.82, pad=0.02)
        cb2.ax.tick_params(labelsize=16)
    _apply_view(ax2, view)

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.93, wspace=0.02)
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.show()
    print(f"Saved plot: {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare orientation norms: dual quaternion error + ln vs reduced+yaw orientation error."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV with columns: t, qd_w,qd_x,qd_y,qd_z,q_w,q_x,q_y,q_z",
    )
    parser.add_argument("--samples", type=int, default=600, help="Samples for demo mode.")
    parser.add_argument(
        "--scenario",
        choices=["demo", "large_to_zero", "angle_sweep", "translation_sweep", "translation_orientation_3d"],
        default="demo",
        help="demo: oscillatory example, large_to_zero: large initial error converging to zero",
    )
    parser.add_argument(
        "--initial-angle-deg",
        type=float,
        default=170.0,
        help="Initial orientation mismatch in degrees for large_to_zero scenario.",
    )
    parser.add_argument(
        "--translation-rot-deg",
        type=float,
        default=120.0,
        help="Constant orientation mismatch (deg) used in translation_sweep to highlight Jacobian effects.",
    )
    parser.add_argument("--rot-axis-x", type=float, default=1.0, help="Rotation axis x component.")
    parser.add_argument("--rot-axis-y", type=float, default=0.0, help="Rotation axis y component.")
    parser.add_argument("--rot-axis-z", type=float, default=0.0, help="Rotation axis z component.")
    parser.add_argument(
        "--view",
        choices=["iso", "top", "side", "front"],
        default="iso",
        help="Camera view for 3D translation_orientation_3d plot.",
    )
    parser.add_argument(
        "--plot-style",
        choices=["surface", "wireframe"],
        default="surface",
        help="3D rendering style for translation_orientation_3d plot.",
    )
    parser.add_argument("--out", type=Path, default=Path("orientation_cost_comparison.png"))
    args = parser.parse_args()

    x_label = "t"
    rot_axis = np.array([args.rot_axis_x, args.rot_axis_y, args.rot_axis_z], dtype=float)
    if args.scenario == "translation_orientation_3d":
        s_t, theta = generate_translation_orientation_grid(args.samples, args.samples)
        dual_log_norm, classical_norm = compute_translation_metrics_grid(s_t, theta, rot_axis)
        plot_translation_orientation_3d(
            s_t,
            theta,
            dual_log_norm,
            classical_norm,
            args.out,
            view=args.view,
            style=args.plot_style,
        )
        return
    elif args.scenario == "translation_sweep":
        x, td, t = generate_translation_to_origin_data(args.samples)
        qd_const = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        q_const = axis_angle_to_quat(rot_axis, math.radians(args.translation_rot_deg))
        qd = np.tile(qd_const, (args.samples, 1))
        q = np.tile(q_const, (args.samples, 1))
        classical, log_dual = compute_translation_metrics(td, t, qd, q)
        plot_translation_results(x, classical, log_dual, args.out)
        return
    elif args.csv is not None:
        t, qd, q = load_csv(args.csv)
    elif args.scenario == "large_to_zero":
        t, qd, q = generate_large_to_zero_data(args.samples, args.initial_angle_deg)
    elif args.scenario == "angle_sweep":
        t, qd, q = generate_angle_sweep_data(args.samples)
        x_label = "Error angle [rad]"
    else:
        t, qd, q = generate_demo_data(args.samples)

    ln_cost, reduced_yaw_cost, _ = compute_metrics(t, qd, q)
    plot_results(t, ln_cost, reduced_yaw_cost, args.out, x_label=x_label)


if __name__ == "__main__":
    main()
