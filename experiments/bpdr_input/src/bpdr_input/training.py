"""Phase 3 training command: three conditions, one budget, every run recorded."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import uuid
import numpy as np
import torch
from .examples import MODES, Normalizer, collate, data_status, load_pairs
from .losses import CONDITIONS, LossWeights, objective
from .metrics import evaluate
from .model import CorrectionEncoder, ModelConfig
from .pipeline import checksum

SOURCE_MODULES = ('training', 'losses', 'model', 'examples', 'metrics', 'laboratory', 'interventions')


def predict(model, batch):
    return {view: model(features, batch['padding']) for view, features in batch['inputs'].items()}


def on_device(batch, device):
    """Tensors move to the training device; identifiers stay as they are."""
    def move(value):
        return {key: move(item) for key, item in value.items()} if isinstance(value, dict) else value.to(device) if torch.is_tensor(value) else value
    return move(batch)


def train(pairs_run, condition, mode, output_root, updates=300, warmup=100, seed=339, batch_size=8, learning_rate=1e-3,
          weights=LossWeights(), config=None, device='cpu'):
    if condition not in CONDITIONS:
        raise ValueError(f'Unknown condition: {condition}')
    if device == 'xpu' and not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
        raise ValueError('This PyTorch build has no Intel GPU support; use the environment with the xpu build or --device cpu')
    pairs_run = Path(pairs_run).resolve()
    pairs = load_pairs(pairs_run, mode)
    features = 70 if condition == 'frame_only' else 71
    config = ModelConfig(**dict(asdict(config or ModelConfig()), features=features))
    torch.manual_seed(seed)
    order = np.random.default_rng(seed)
    normalizer = Normalizer.fit(pairs)
    model = CorrectionEncoder(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history, skipped_total = [], {}
    model.train()
    for step in range(updates):
        chosen = order.choice(len(pairs), size=min(batch_size, len(pairs)), replace=False)
        batch = on_device(collate([pairs[i] for i in chosen], normalizer, features), device)
        # Supervised warm-up first; the response terms join afterwards and only for the BPDR condition.
        loss, parts, skipped = objective(predict(model, batch), batch, condition, weights, response=step >= warmup)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        history.append(dict(step=step, loss=loss.item(), **{name: value.item() for name, value in parts.items()}))
        for name, count in skipped.items():
            skipped_total[name] = skipped_total.get(name, 0)+count
    model.eval()
    everything = collate(pairs, normalizer, features)
    with torch.inference_mode():
        fit = evaluate(predict(model, on_device(everything, device)), everything)
    directory = Path(output_root).resolve()/mode/uuid.uuid4().hex
    directory.mkdir(parents=True)
    state = {name: value.cpu() for name, value in model.state_dict().items()}
    # Every saved file carries the data status, so a checkpoint reloaded later cannot pass as reviewed work.
    origin = dict(data_status=data_status(mode), pairs_summary_sha256=checksum(pairs_run/'summary.json'))
    torch.save(dict(state=state, config=asdict(config), normalizer=normalizer.state(), condition=condition, **origin), directory/'checkpoint.pt')
    module_dir = Path(__file__).resolve().parent
    record = dict(**origin, mode=mode, condition=condition, seed=seed, updates=updates, warmup_updates=warmup,
                  batch_size=batch_size, learning_rate=learning_rate, optimizer='AdamW', loss_weights=asdict(weights), device=device,
                  model=model.describe(), pairs=len(pairs), singers=len({pair['singer_id'] for pair in pairs}),
                  pairs_run=str(pairs_run), input_sha256={pair['pair_id']: pair['input_sha256'] for pair in pairs},
                  source_code_sha256={name: checksum(module_dir/(name+'.py')) for name in SOURCE_MODULES},
                  checkpoint_sha256=checksum(directory/'checkpoint.pt'), torch_version=torch.__version__,
                  numpy_version=np.__version__, python_version=sys.version.split()[0], interpreter=sys.executable,
                  empty_mask_exclusions=skipped_total, first_step=history[0] if history else None, last_step=history[-1] if history else None,
                  training_fit=fit['summary'],
                  interpretation='Fit measured on the same pairs the model was trained on; this is not an evaluation result',
                  limitations=['One run and one seed cannot show training stability',
                               'Repeated shifts and boundary variants of one phrase are not independent examples'])
    (directory/'run_record.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    (directory/'history.json').write_text(json.dumps(dict(**origin, steps=history)), encoding='utf-8')
    (directory/'training_fit.json').write_text(json.dumps(dict(**origin, **fit), indent=2), encoding='utf-8')
    return directory, record


def reload(directory):
    """Rebuild a saved model and its normalizer as they were at the end of training, with the status of its data."""
    saved = torch.load(Path(directory)/'checkpoint.pt', weights_only=True)
    model = CorrectionEncoder(ModelConfig(**saved['config']))
    model.load_state_dict(saved['state'])
    return model.eval(), Normalizer(**saved['normalizer']), saved['condition'], saved['data_status']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    component = Path(__file__).resolve().parents[2]
    parser.add_argument('--pairs-run', type=Path, required=True, help='Phase 2 experiment directory containing summary.json')
    parser.add_argument('--condition', choices=CONDITIONS, required=True)
    parser.add_argument('--mode', choices=MODES, default='research', help='engineering reads unreviewed audited pairs and marks every output exploratory')
    parser.add_argument('--output-root', type=Path, default=component/'runs')
    parser.add_argument('--updates', type=int, default=300)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--seed', type=int, default=339)
    parser.add_argument('--device', choices=('cpu', 'xpu', 'cuda'), default='cpu', help='xpu is an Intel GPU and needs the xpu build of PyTorch')
    args = parser.parse_args()
    directory, record = train(args.pairs_run, args.condition, args.mode, args.output_root, args.updates, args.warmup, args.seed, device=args.device)
    print(json.dumps({key: record[key] for key in ('data_status', 'condition', 'device', 'pairs', 'singers', 'model', 'last_step', 'training_fit')}, indent=2))
    print('TRAINING_RUN:', directory)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
