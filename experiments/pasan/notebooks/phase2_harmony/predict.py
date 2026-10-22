"""CLI for the same inference function demonstrated in notebook 04."""
import argparse
import json
from pathlib import Path

from .model import load_bundle, harmonize


def main():
    parser=argparse.ArgumentParser(description='Normalized melody + genre + mode to Roman chords.')
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    model,metadata,transition=load_bundle(args.bundle)
    data=json.loads(args.input.read_text())
    result=harmonize(model,notes=data['notes'],beat_boundaries=data['beat_boundaries'],
                     mode=data['mode'],genres=data.get('genres',[]),genre_vocab=metadata['genre_vocab'],
                     tonic_pc=data.get('tonic_pc'),normalized=data.get('normalized',False),
                     beats=data.get('beats'),transition=transition,alpha=metadata['alpha'])
    result.pop('logits')
    text=json.dumps(result,indent=2)
    if args.output:
        args.output.write_text(text+'\n')
    else:
        print(text)


if __name__=='__main__':
    main()
