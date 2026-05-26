#!/usr/bin/env python3
"""
Run ``bench_fwd.py`` for each requested varlen metadata mode at default ``H`` and
``--H 64``, parse stdout, and write a benchmark markdown report.

Reports mean latency for ``flash_kda (fp32 state)`` and ``fla_chunk_kda`` (FLA
``chunk_kda``), plus speedup ``chunk_mean / flash_mean``. Generated date is UTC,
day precision only (YYYY-MM-DD).
"""
from __future__ import annotations

import argparse
import ast
import datetime as _dt
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_FWD = Path(__file__).resolve().parent / "bench_fwd.py"
DEFAULT_OUT = REPO_ROOT / "BENCHMARK_H20.md"
DEFAULT_DEVICE_LABEL = "Hopper / H20"

# Matches ``chunk_kda(...)`` in ``benchmarks/bench_fwd.py`` (documented in the report).
FLA_CHUNK_KDA_OPTIONS_MD = (
    "- `fla_chunk_kda` configuration: `use_gate_in_kernel=True`, "
    "`use_qk_l2norm_in_kernel=True`, `use_beta_sigmoid_in_kernel=True`, "
    "`lower_bound=-5`, `transpose_state_layout=True`"
)
FLA_CHUNK_GDN_OPTIONS_MD = (
    "- `fla_chunk_gated_delta_rule` configuration: scalar per-head gate "
    "`g` of shape `(1, T, H)`, `use_qk_l2norm_in_kernel=True`, "
    "`transpose_state_layout=True`"
)

RE_HEADER_FIXED = re.compile(
    r"^shape=\[(\d+),(\d+),(\d+)\] warmup=(\d+) iters=(\d+) repeats=(\d+)\s*$"
)
RE_HEADER_VARLEN = re.compile(
    r"^varlen shape=\[(\d+),(\d+),(\d+)\] seq_lens=(.+?) "
    r"(?:use_varlen_metadata=(\w+) )?"
    r"warmup=(\d+) iters=(\d+) repeats=(\d+)\s*$"
)
RE_RESULT = re.compile(
    r"^\s+(.+?)\s*:\s*mean=([\d.]+) ms, min=([\d.]+) ms, max=([\d.]+) ms\s*$"
)


def run_bench(extra_argv: list[str]) -> str:
    cmd = [sys.executable, str(BENCH_FWD), *extra_argv]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or "")
        sys.stderr.write(proc.stdout or "")
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)
    return proc.stdout


def parse_stdout(text: str) -> list[dict]:
    """
    Each case: kind, T,H,D, warmup, iters, repeats, seq_lens (varlen), optional
    flash_mean_ms, chunk_mean_ms (floats).
    """
    cases: list[dict] = []
    current: dict | None = None

    def new_case_base(
        kind: str,
        *,
        T: int,
        H: int,
        D: int,
        warmup: int,
        iters: int,
        repeats: int,
        seq_lens: str | None = None,
    ) -> dict:
        c: dict = {
            "kind": kind,
            "T": T,
            "H": H,
            "D": D,
            "warmup": warmup,
            "iters": iters,
            "repeats": repeats,
            "flash_mean_ms": None,
            "chunk_mean_ms": None,
            "gdn_mean_ms": None,
        }
        if seq_lens is not None:
            c["seq_lens"] = seq_lens
        return c

    for line in text.splitlines():
        m = RE_HEADER_VARLEN.match(line)
        if m:
            if current is not None:
                cases.append(current)
            t, h, d, seq_lens, varlen_metadata, w, it, rep = m.groups()
            current = new_case_base(
                "varlen",
                T=int(t),
                H=int(h),
                D=int(d),
                warmup=int(w),
                iters=int(it),
                repeats=int(rep),
                seq_lens=seq_lens,
            )
            current["varlen_metadata"] = varlen_metadata or "default"
            continue

        m = RE_HEADER_FIXED.match(line)
        if m:
            if current is not None:
                cases.append(current)
            t, h, d, w, it, rep = m.groups()
            current = new_case_base(
                "fixed",
                T=int(t),
                H=int(h),
                D=int(d),
                warmup=int(w),
                iters=int(it),
                repeats=int(rep),
            )
            continue

        m = RE_RESULT.match(line)
        if m and current is not None:
            name, mean, _mn, _mx = m.groups()
            name = name.strip()
            if "fp32 state" in name:
                current["flash_mean_ms"] = float(mean)
            elif name == "chunk_kda":
                current["chunk_mean_ms"] = float(mean)
            elif name == "chunk_gated_delta_rule":
                current["gdn_mean_ms"] = float(mean)

    if current is not None:
        cases.append(current)

    return cases


def _fmt_seq_lens(seq_lens_str: str) -> str:
    """Uniform segment lengths become ``1024 x 8``; mixed lists keep the bracket form."""
    m = re.fullmatch(r"\[(\d+)\]\s*\*\s*(\d+)", seq_lens_str)
    if m:
        return f"{m.group(1)} x {m.group(2)}"
    try:
        xs = ast.literal_eval(seq_lens_str)
    except (ValueError, SyntaxError):
        return seq_lens_str
    if not isinstance(xs, list) or not xs:
        return seq_lens_str
    if not all(isinstance(x, int) for x in xs):
        return seq_lens_str
    first = xs[0]
    if len(xs) >= 2 and all(x == first for x in xs):
        return f"{first} x {len(xs)}"
    return seq_lens_str


def _case_detail(c: dict) -> str:
    """Scenario text for the Case column (T/H/D are printed once above the table)."""
    if c["kind"] == "fixed":
        return "Fixed"
    seq = _fmt_seq_lens(c["seq_lens"])
    if seq.startswith("["):
        return f"Varlen, `seq_lens`={seq}"
    return f"Varlen, `seq_lens`=`{seq}`"


def _fmt_ms(x: float) -> str:
    return f"{x:.4f}"


def _fmt_speedup(flash: float, chunk: float) -> str:
    if flash <= 0:
        return "—"
    return f"{chunk / flash:.2f}×"


def _argv_with_h(argv: list[str], h: int) -> list[str]:
    """Drop any ``--H`` / ``--H=`` from *argv*, then append ``--H`` *h*."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--H" and i + 1 < len(argv):
            i += 2
            continue
        if a.startswith("--H="):
            i += 1
            continue
        out.append(a)
        i += 1
    out.extend(["--H", str(h)])
    return out


def _argv_with_varlen_metadata(argv: list[str], mode: str) -> list[str]:
    """Drop any ``--use-varlen-metadata`` from *argv*, then append *mode*."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--use-varlen-metadata" and i + 1 < len(argv):
            i += 2
            continue
        if a.startswith("--use-varlen-metadata="):
            i += 1
            continue
        out.append(a)
        i += 1
    out.extend(["--use-varlen-metadata", mode])
    return out


def _complete_cases(raw: list[dict]) -> list[dict]:
    return [
        c
        for c in raw
        if c.get("flash_mean_ms") is not None
        and c.get("chunk_mean_ms") is not None
        and c.get("gdn_mean_ms") is not None
    ]


def _render_table_block(cases: list[dict]) -> list[str]:
    lines: list[str] = [
        "| Case | `flash_kda` mean (ms) | `fla_chunk_kda` mean (ms) | "
        "Speedup vs `chunk_kda` | `fla_chunk_gdn` mean (ms) | "
        "Speedup vs `gdn` |",
        "|------|----------------------:|----------------------:|--------:|"
        "----------------------:|--------:|",
    ]
    for c in cases:
        flash = c["flash_mean_ms"]
        chunk = c["chunk_mean_ms"]
        gdn = c["gdn_mean_ms"]
        cell = _case_detail(c).replace("|", "\\|")
        lines.append(
            f"| {cell} | {_fmt_ms(flash)} | {_fmt_ms(chunk)} |"
            f" {_fmt_speedup(flash, chunk)} |"
            f" {_fmt_ms(gdn)} |"
            f" {_fmt_speedup(flash, gdn)} |"
        )
    lines.append("")
    return lines


def render_markdown(
    sections: list[tuple[str, list[dict]]],
    generated_at: str,
    generator_cmd: str,
    device_label: str,
) -> str:
    """
    *sections*: ``(label, cases)`` pairs, one per table.
    *generator_cmd*: command that reproduces this report.
    *device_label*: device/platform label printed in the report title.
    """
    title = "# KDA forward benchmark"
    if device_label:
        title += f" ({device_label})"

    lines: list[str] = [
        title,
        "",
        f"- Generated: {generated_at}",
        "",
    ]

    if not sections:
        lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    lines.append(f"- Command: `{generator_cmd}`")
    lines.append("")

    first_cases = next((cases for _label, cases in sections if cases), None)
    c0 = first_cases[0] if first_cases else None
    if c0 is not None:
        lines.append(
            f"- Benchmark settings: `warmup={c0['warmup']}`, `iters={c0['iters']}`, "
            f"`repeats={c0['repeats']}`"
        )
        lines.append("")
        lines.append(FLA_CHUNK_KDA_OPTIONS_MD)
        lines.append(FLA_CHUNK_GDN_OPTIONS_MD)
        lines.append("")

    for label, cases in sections:
        if not cases:
            continue
        c0 = cases[0]
        title = f"`T={c0['T']}`, `H={c0['H']}`, `D={c0['D']}`"
        if label:
            title += f", `{label}`"
        lines.append(f"### {title}")
        lines.append("")
        lines.extend(_render_table_block(cases))

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run bench_fwd.py and write a benchmark markdown report."
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output markdown path (default: {DEFAULT_OUT})",
    )
    p.add_argument(
        "--device-label",
        default=DEFAULT_DEVICE_LABEL,
        help=f"Device/platform label for the report title (default: {DEFAULT_DEVICE_LABEL!r})",
    )
    p.add_argument(
        "--varlen-metadata-modes",
        default="default,off,on",
        help=(
            "Comma-separated --use-varlen-metadata modes to benchmark "
            "(default: default,off,on)."
        ),
    )
    args, bench_extra = p.parse_known_args()
    metadata_modes = [m.strip() for m in args.varlen_metadata_modes.split(",") if m.strip()]
    invalid_modes = sorted(set(metadata_modes) - {"default", "on", "off"})
    if invalid_modes:
        p.error(f"invalid --varlen-metadata-modes value(s): {', '.join(invalid_modes)}")

    def _fmt_generator_cmd(extra: list[str]) -> str:
        cmd = "python benchmarks/generate_benchmark_md.py"
        if args.output != DEFAULT_OUT:
            cmd += f" -o {args.output}"
        if args.device_label != DEFAULT_DEVICE_LABEL:
            cmd += f" --device-label {args.device_label}"
        if args.varlen_metadata_modes != "default,off,on":
            cmd += f" --varlen-metadata-modes {args.varlen_metadata_modes}"
        tail = " ".join(extra)
        return f"{cmd} {tail}".strip() if tail else cmd

    sections: list[tuple[str, list[dict]]] = []
    for mode in metadata_modes:
        argv_mode = _argv_with_varlen_metadata(bench_extra, mode)
        for h in (None, 64):
            argv = list(argv_mode) if h is None else _argv_with_h(argv_mode, h)
            stdout = run_bench(argv)
            cases = _complete_cases(parse_stdout(stdout))
            label = f"use_varlen_metadata={mode}"
            sections.append((label, cases))

    if any(not cases for _label, cases in sections):
        sys.stderr.write(
            "Warning: missing complete benchmark rows for one or more runs "
            "(need fp32 state, fla_chunk_kda, and fla_chunk_gated_delta_rule "
            "for each).\n"
        )

    generated = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    md = render_markdown(sections, generated, _fmt_generator_cmd(bench_extra), args.device_label)
    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
