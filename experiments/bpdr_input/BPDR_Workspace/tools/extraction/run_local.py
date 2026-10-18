"""Run the mirrored melody-extraction pipeline unmodified on this machine.

The pipeline in data/mtg-jamendo/melody-extraction-pipeline is the team's code, written for an
Apple Silicon Mac. Nothing in it is edited: this launcher supplies what another platform lacks,
checks the model tools, and then calls the pipeline's own functions.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types

COMPONENT = next(p for p in Path(__file__).resolve().parents if p.name == 'bpdr_input' and (p/'pyproject.toml').is_file())
PIPELINE = COMPONENT/'data'/'mtg-jamendo'/'melody-extraction-pipeline'
ASSETS = COMPONENT/'checkpoints'/'melody-extraction'
STEMS = COMPONENT/'data'/'jamendo-vocal-stems'


def platform_shims(overlap=None, device=None):
    """Return the substitutions made; an empty list means the pipeline runs natively."""
    used = []
    if not installed('fcntl'):
        import msvcrt
        shim = types.ModuleType('fcntl')
        shim.LOCK_EX, shim.LOCK_NB = 2, 4

        def flock(stream, flags):
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise BlockingIOError(str(exc)) from exc
        shim.flock = flock
        sys.modules['fcntl'] = shim
        used.append('run lock: msvcrt.locking instead of fcntl.flock')
    if not installed('mlx_audio_separator'):
        shim = types.ModuleType('mlx_audio_separator')

        def __getattr__(name):
            if name != 'Separator':
                raise AttributeError(name)
            # Same constructor, load_model and separate calls; imported only when a stem is needed.
            from audio_separator.separator import Separator
            if overlap is None and device is None:
                return Separator
            import inspect
            import torch

            def configured(**options):
                if overlap is not None:
                    options['mdxc_params'] = dict(inspect.signature(Separator.__init__).parameters['mdxc_params'].default, overlap=overlap)
                separator = Separator(**options)
                if device is not None:
                    # The package selects only CUDA, Apple or DirectML devices itself; an Intel GPU is set here.
                    separator.torch_device = torch.device(device)
                return separator
            return configured
        shim.__getattr__ = __getattr__
        sys.modules['mlx_audio_separator'] = shim
        used.append('vocal separator: audio-separator (PyTorch) instead of mlx-audio-separator, same BS-RoFormer checkpoint')
        if overlap is not None:
            used.append(f'vocal separator: chunk overlap {overlap} instead of the model default 4; faster, stems differ from the default')
        if device is not None:
            used.append(f'vocal separator: runs on PyTorch device {device} instead of the CPU; same model and settings')
    elif overlap is not None or device is not None:
        raise SystemExit('--separator-overlap and --device apply only to the PyTorch separator substitution')
    return used


def load_pipeline(root):
    spec = importlib.util.spec_from_file_location('run', root/'run.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules['run'] = module
    spec.loader.exec_module(module)
    return module


def installed(name):
    try:
        return importlib.util.find_spec(name) is not None
    except ValueError:  # a shim placed in sys.modules has no spec
        return False


def missing_tools(module, vocal_only):
    """A missing tool would be recorded by the pipeline as a failed song, so stop before it starts."""
    assets, problems = module.ASSETS, []
    problems += [f'{tool} is not on PATH' for tool in ('ffmpeg', 'ffprobe') if not shutil.which(tool)]
    for name in ['.venv-game'] + ([] if vocal_only else ['.venv-sheetsage2']):
        # The pipeline hard-codes the POSIX layout; on Windows bin is a junction to Scripts.
        if not any((assets/name/'bin'/executable).exists() for executable in ('python', 'python.exe')):
            problems.append(f'{name}/bin/python is missing')
    required = [module.GAME_ROOT/'infer.py', module.GAME_MODEL]
    if not vocal_only:
        required += [assets/'models'/'hf-home'/'hub'/name for name in ('models--m-a-p--SheetSage2', 'models--m-a-p--MERT-v2-FullSong')]
    problems += [f'{path} is missing' for path in required if not path.exists()]
    try:
        # Import it now: a separator that cannot load would otherwise fail every selected song.
        from mlx_audio_separator import Separator  # noqa: F401
    except ImportError as exc:
        problems.append(f'the vocal separator cannot be imported: {exc}')
    problems += [f'{name} is not installed in this Python environment' for name in ('pretty_midi', 'soundfile')
                 if not installed(name)]
    return problems


def keep_stems(module, directory):
    """BPDR needs the separated vocal audio, which the pipeline deletes after note extraction."""
    directory.mkdir(parents=True, exist_ok=True)
    separate = module.GameExtractor.stem

    def stem(self, track, seconds):
        path = separate(self, track, seconds)
        kept = directory/path.name
        if not kept.exists():
            try:
                os.link(path, kept)
            except OSError:
                shutil.copyfile(path, kept)
        return path
    module.GameExtractor.stem = stem


def queue_first(module, wanted):
    """The pipeline takes songs in inventory order; put the named ones at the front for this run."""
    inventory = module.inventory

    def reordered():
        tracks = inventory()
        unknown = sorted(set(wanted)-{track['track_id'] for track in tracks})
        if unknown:
            raise SystemExit(f'Not in the local inventory: {unknown}')
        return sorted(tracks, key=lambda track: wanted.index(track['track_id']) if track['track_id'] in wanted else len(wanted))
    module.inventory = reordered


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog='Other options, such as --limit 2 and --retry-failed, go to the pipeline unchanged.')
    parser.add_argument('--pipeline', type=Path, default=PIPELINE)
    parser.add_argument('--first', help='Comma-separated track IDs to process before the rest; sets --limit to their count if none is given')
    parser.add_argument('--index-only', action='store_true', help='Write manifest.csv and catalog.json only; needs no model')
    parser.add_argument('--vocal-only', action='store_true', help='Skip SheetSage2; songs stay partial with GAME vocal notes')
    parser.add_argument('--keep-stems', action='store_true', help=f'Keep each separated vocal WAV in {STEMS.relative_to(COMPONENT)}')
    parser.add_argument('--threads', type=int, help='CPU threads for the separator; changes speed, not the model')
    parser.add_argument('--separator-overlap', type=int, choices=(2, 3, 4), help='Fewer overlapping chunks: faster separation, slightly different stems')
    parser.add_argument('--device', choices=('xpu',), help='Run the separator on an Intel GPU; needs the xpu build of PyTorch')
    args, forwarded = parser.parse_known_args()
    if os.name == 'nt' and not sys.flags.utf8_mode:
        # The pipeline writes text with the platform encoding; match the Mac's UTF-8.
        return subprocess.call([sys.executable, '-X', 'utf8', *sys.argv])
    os.environ['PYTHONUTF8'] = '1'
    os.environ.setdefault('MELODY_MODEL_ASSETS', str(ASSETS))
    local_ffmpeg = Path(os.environ['MELODY_MODEL_ASSETS'])/'tools'/'ffmpeg'/'bin'
    if local_ffmpeg.is_dir():
        os.environ['PATH'] = str(local_ffmpeg)+os.pathsep+os.environ['PATH']
    shims = platform_shims(args.separator_overlap, args.device)
    root = args.pipeline.resolve()
    if not (root/'run.py').exists():
        parser.error(f'No pipeline at {root}. Run Setup-Jamendo.ps1 first.')
    module = load_pipeline(root)
    if args.index_only:
        tracks = module.inventory()
        errors, durations = module.prior_fields()
        module.publish_index(tracks, errors, durations)
        print(f'Indexed {len(tracks)} local MP3s: {module.MANIFEST}')
        return 0
    problems = missing_tools(module, args.vocal_only)
    if args.device == 'xpu':
        import torch
        if not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
            problems.append('no Intel GPU is visible to this PyTorch build; use the .venv-xpu environment')
    if problems:
        print('Extraction tools are incomplete; nothing was run:\n  '+'\n  '.join(problems), file=sys.stderr)
        return 2
    if args.vocal_only:
        module.run_sheet_queue = lambda queue, *rest: print(
            f'SheetSage2 skipped for {len(queue)} songs (--vocal-only); they stay partial.', flush=True)
    if args.keep_stems:
        keep_stems(module, STEMS)
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    if args.first:
        wanted = [value.strip() for value in args.first.split(',') if value.strip()]
        queue_first(module, wanted)
        if '--limit' not in forwarded:
            forwarded += ['--limit', str(len(wanted))]
    for shim in shims:
        print('Local substitution -', shim, flush=True)
        module.log('*', 'local_platform', shim)
    sys.argv = [str(root/'run.py'), *forwarded]
    try:
        module.main()
    except KeyboardInterrupt:
        module.log('*', 'interrupted', 'Ctrl-C; rerun the same command to continue')
        print('Stopped. Completed files are saved; rerun the same command to continue.', flush=True)
        return 130
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
