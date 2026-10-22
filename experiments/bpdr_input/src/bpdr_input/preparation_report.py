"""Local listening/index page for a verified preparation run; cannot approve references."""
import argparse
from html import escape
import json
from pathlib import Path
from urllib.parse import quote
from .data_builder import validate_run


def render(run):
    run = Path(run).resolve()
    validation = validate_run(run)
    manifest = json.loads((run/'manifest.json').read_text())
    summary = json.loads((run/'summary.json').read_text())
    def link(path):
        relative = Path(path).resolve().relative_to(run).as_posix()
        if not (run/relative).is_file():
            raise FileNotFoundError(relative)
        return quote(relative, safe='/')
    cards = []
    for record in manifest['records']:
        screen = record['screen']
        content = '<h2>'+escape(record['record_id'])+'</h2>'
        content += '<p>'+escape(record['dataset'])+' | '+escape(record['reference_kind'])+' | '+escape(record['group_kind'])+'</p>'
        content += '<p><strong>'+escape(screen['status'].replace('_', ' '))+'</strong>: '+escape(', '.join(screen['reasons']) or 'Automatic screen passed; independent musical review is still required.')+'</p>'
        if 'accepted_audio_path' in record:
            content += '<audio controls preload="none" src="'+link(record['accepted_audio_path'])+'"></audio>'
            content += '<p>Source crop begins at '+str(round(record['source_map']['original_offset_seconds'], 3))+' seconds. Duration '+str(round(record['duration_seconds'], 3))+' seconds; no musical repair.</p>'
            technical = Path(record['diagnostics_path']).parent/'report.html'
            content += '<p><a href="'+link(technical)+'">Original and accepted audio with acoustic plots</a> · <a href="'+link(record['model_notes']['path'])+'">Model-estimated notes</a></p>'
            if record['reference_kind'] == 'human_annotation':
                content += '<p><a href="'+link(record['reference_files']['notes']['path'])+'">Human A1/A2 references</a> · <a href="'+link(record['reference_files']['f0']['path'])+'">Human F0 reference</a></p>'
            else:
                content += '<p>No human pitch/note annotation. '+('Legacy stem settings were not individually recorded.' if 'legacy' in record['separation_provenance'] else 'A source-matched separation receipt is retained.')+'</p>'
            content += '<details><summary>Screen measurements</summary><pre>'+escape(json.dumps(screen, indent=2))+'</pre></details>'
        cards.append('<article>'+content+'</article>')
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BPDR preparation review</title><style>body{font:17px/1.6 system-ui;background:#f4f6f8;color:#17212c;max-width:1040px;margin:36px auto;padding:0 20px}h1{line-height:1.2}article{background:white;border:1px solid #d7dde3;border-radius:10px;padding:22px;margin:20px 0}audio{width:100%}pre{white-space:pre-wrap;font-size:13px}.notice{border-left:4px solid #ad6d00;padding:12px 18px;background:#fff5dd}a{color:#145e99}</style>
<h1>BPDR preparation: listen and inspect</h1><p class="notice">Technical acceptance and automatic screening are not musical approval. These are exploratory preparation records: <strong>zero research-ready examples</strong>. Two independent reviews and realized-shift audits remain required.</p>
<p><a href="reviews.csv">Pending review sheet</a> · <a href="manifest.json">Provenance manifest</a> · <a href="summary.json">Run summary</a></p>'''
    html += '<p>'+str(summary['records'])+' crops; '+str(summary['review_rows'])+' pending review rows. Source files and accepted packages verified.</p>'
    html += ''.join(cards)+'</html>'
    (run/'review.html').write_text(html, encoding='utf-8')
    return dict(path=str(run/'review.html'), validation=validation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.run), indent=2))


if __name__ == '__main__':
    main()
