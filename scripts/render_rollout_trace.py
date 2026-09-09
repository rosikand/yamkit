"""Render saved trace artifacts only; never imports or connects robot hardware."""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import stat
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIDES = ('left', 'right')
JOINTS = tuple(f'joint_{i}.pos' for i in range(1, 7)) + ('gripper.pos',)
MAX_JSON_BYTES = 16 * 1024 * 1024


def repo_path(path, *, directory=False):
    path = Path(path).absolute()
    root = ROOT.resolve()
    if '..' in path.parts or not path.is_relative_to(root):
        raise ValueError('Trace paths must remain inside this repository')
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            raise ValueError('Trace paths cannot contain symlinks')
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ValueError('Trace paths must remain inside this repository')
    if directory and not resolved.is_dir():
        raise ValueError('Trace directory must already exist')
    return resolved


def read_json(path):
    path = repo_path(path)
    # O_NONBLOCK also prevents a substituted FIFO from hanging this offline reader.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_JSON_BYTES:
            raise ValueError('Trace JSON must be a bounded regular file')
        raw = stream.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError('Trace JSON exceeded its bound while being read')
    def reject(value):
        raise ValueError('Trace JSON contains a non-finite number')
    value = json.loads(raw, parse_constant=reject)
    def finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError('Trace JSON contains a non-finite number')
        if isinstance(item, list):
            for part in item:
                finite(part)
        elif isinstance(item, dict):
            for part in item.values():
                finite(part)
    finite(value)
    return value


def series(summary, trace):
    if not isinstance(summary, dict) or not isinstance(trace, dict):
        raise ValueError('Trace and summary must be JSON objects')  # noqa: TRY004 — uniform malformed-JSON boundary.
    start = summary.get('phase_started_monotonic_s')
    if type(start) not in (int, float) or not math.isfinite(start):
        raise ValueError('A recorded policy phase start is required')
    events = trace.get('events')
    if not isinstance(events, list) or len(events) > 4096:
        raise ValueError('Trace event count exceeds its bound')
    expected = [f'{side}_{joint}' for side in SIDES for joint in JOINTS]
    if summary.get('action_names') != expected:
        raise ValueError('Trace action ordering does not match bimanual YAM')
    chunks = trace.get('chunks', [])
    if not isinstance(chunks, list) or len(chunks) > 128:
        raise ValueError('Trace chunk count exceeds its bound')
    for chunk in chunks:
        actions = chunk.get('actions') if isinstance(chunk, dict) else None
        if (not isinstance(actions, list) or len(actions) != 30
                or any(not isinstance(row, list) or len(row) != 14 for row in actions)
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       for row in actions for value in row)):
            raise ValueError('Policy chunks must contain finite 30 by 14 targets')
    result = {side: {joint: {'measured': [], 'requested': [], 'sent': []} for joint in JOINTS} for side in SIDES}
    merges, errors = [], []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError('Trace events must be JSON objects')  # noqa: TRY004 — uniform malformed-JSON boundary.
        at = event.get('monotonic_s')
        if type(at) not in (int, float) or not math.isfinite(at):
            raise ValueError('Trace event timestamp is invalid')
        t = at - start
        kind = event.get('kind')
        if kind == 'chunk_merge':
            merges.append(t)
        elif kind == 'send_error':
            errors.append(t)
        elif kind == 'observation':
            values = event.get('positions')
            if not isinstance(values, list) or len(values) != 14:
                raise ValueError('Measured joint vector must contain 14 values')
            for side_index, side in enumerate(SIDES):
                for index, joint in enumerate(JOINTS):
                    value = values[side_index * 7 + index]
                    if type(value) not in (int, float) or not math.isfinite(value):
                        raise ValueError('Measured joint position is invalid')
                    result[side][joint]['measured'].append((t, value))
        elif kind in ('send_start', 'send_end'):
            arm = event.get('arm')
            if arm not in ('left_follower', 'right_follower'):
                raise ValueError('Unexpected trace arm identity')
            side = arm.removesuffix('_follower')
            field = 'requested' if kind == 'send_start' else 'postclamp'
            key = 'requested' if kind == 'send_start' else 'sent'
            values = event.get(field)
            if not isinstance(values, dict) or set(values) != set(JOINTS):
                raise ValueError('Command trace must contain all seven side targets')
            for joint, value in values.items():
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError('Command target is invalid')
                result[side][joint][key].append((t, value))
    return result, merges, errors


def render(directory):
    directory = repo_path(directory, directory=True)
    # Check every destination before rendering anything. Atomic replacement below
    # never follows an output symlink even if one appears after this check.
    for name in ('report.html', 'joints-left.png', 'joints-right.png'):
        destination = repo_path(directory / name)
        if destination.exists() and not destination.is_file():
            raise ValueError('Trace output must be a regular file')
    summary = read_json(directory / 'summary.json')
    trace = read_json(directory / 'trace.json')
    values, merges, errors = series(summary, trace)
    cache = repo_path(ROOT / '.context' / 'matplotlib')
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(cache)
    import matplotlib
    if any(not Path(getter()).resolve().is_relative_to(ROOT.resolve())
           for getter in (matplotlib.get_cachedir, matplotlib.get_configdir)):
        raise ValueError('Use a fresh renderer process with a repository-local Matplotlib cache')
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    colors = {'measured': '#111827', 'requested': '#d97706', 'sent': '#2563eb'}
    labels = {'measured': 'Measured position', 'requested': 'Policy target', 'sent': 'Target after yamkit clamp'}
    for side in SIDES:
        fig, axes = plt.subplots(7, 1, figsize=(12, 15), sharex=True, constrained_layout=True)
        try:
            for joint, ax in zip(JOINTS, axes, strict=True):
                for key in ('requested', 'sent', 'measured'):
                    points = values[side][joint][key]
                    if points:
                        x, y = zip(*points, strict=True)
                        # Points only: missing data and failed sends are never interpolated.
                        ax.plot(x, y, linestyle='none', marker='.', markersize=3,
                                color=colors[key], alpha=.75, label=labels[key])
                for at in merges:
                    ax.axvline(at, color='#9ca3af', linewidth=.6, alpha=.45)
                for at in errors:
                    ax.axvline(at, color='#dc2626', linewidth=1, linestyle='--')
                ax.set_ylabel(joint.replace('.pos', '') + (' [0–1]' if joint == 'gripper.pos' else ' [rad]'))
                ax.grid(alpha=.15)
            axes[0].legend(loc='upper left', ncol=3, fontsize=8)
            axes[-1].set_xlabel('Seconds from policy phase start (local receipt/dispatch timestamps)')
            fig.suptitle(f'{side.title()} follower · measured and commanded trajectories')
            atomic_output(directory / f'joints-{side}.png', lambda path, fig=fig: fig.savefig(path, dpi=120))
        finally:
            plt.close(fig)
    videos = ''.join(f'<figure><figcaption>{name}</figcaption><video controls preload="metadata" src="{name}.mp4"></video></figure>'
                     for name in ('top', 'left_wrist', 'right_wrist')
                     if not (directory / f'{name}.mp4').is_symlink() and (directory / f'{name}.mp4').is_file())
    counts = html.escape(json.dumps(summary.get('counts', {}), sort_keys=True))
    title = 'Synthetic software fixture' if summary.get('synthetic_fixture') is True else 'Saved rollout trace'
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{title}</title><style>body{{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#111827}}img{{width:100%}}video{{width:100%;max-width:640px}}figure{{margin:16px 0}}code{{overflow-wrap:anywhere}}</style>
<h1>{title}</h1><p>{'Generated test data only. No arms were connected; this is not evidence of a physical rollout.' if summary.get('synthetic_fixture') is True else ''}</p><p>{html.escape(str(summary.get('task', '')))}</p>
<p>Gray vertical lines mark chunk merges; red dashed lines mark failed sends. Targets after the yamkit clamp are commands, not proof of movement. SDK gripper force limiting may modify the gripper target further.</p>
<p>Points show recorded samples only. Camera exposure timestamps are unavailable. Video is sampled at up to 5 fps; consult frame_timestamps.json for gaps and exact receipt times. These plots do not establish the cause of jitter or task success.</p>
<p>Resources released: {html.escape(str(summary.get('resources_released')))}. Overflow: {html.escape(str(summary.get('overflow')))}. Trace counters: <code>{counts}</code>.</p>
<h2>Video</h2>{videos or '<p>No video is available.</p>'}
<h2>Left follower</h2><img alt="Left follower joint traces" src="joints-left.png">
<h2>Right follower</h2><img alt="Right follower joint traces" src="joints-right.png">
<p><a href="summary.json">Summary</a> · <a href="trace.json">Raw trajectories and chunks</a> · <a href="frame_timestamps.json">Frame timestamps</a> · <a href="metrics.json">Full metrics</a></p></html>'''
    atomic_output(directory / 'report.html', lambda path: path.write_text(document))
    return directory / 'report.html'


def atomic_output(destination, write):
    destination = repo_path(destination)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.trace-render-',
                                      suffix=destination.suffix, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        write(temporary)
        repo_path(destination)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(render(args.directory))


if __name__ == '__main__':
    main()
