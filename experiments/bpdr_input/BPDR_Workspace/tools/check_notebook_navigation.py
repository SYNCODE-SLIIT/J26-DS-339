"""Validate local links/anchors and prepare an offline search-script fixture."""
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import uuid
from urllib.parse import unquote,urlsplit

WORKSPACE=Path(__file__).resolve().parents[1]


class Page(HTMLParser):
    def __init__(self,text):
        super().__init__(convert_charrefs=True)
        self.ids=set();self.links=[];self.cards=[];self.copy_ids=[];self.code={};self.script='';self.pre=None;self.in_script=False
        self.feed(text)
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        if 'id' in d:self.ids.add(d['id'])
        if tag=='a' and 'href' in d:self.links.append(d['href'])
        if tag=='article' and 'data-search' in d:self.cards.append(dict(id=d['id'],dataset=dict(search=d['data-search'],kind=d['data-kind'])))
        if tag=='button' and 'data-copy' in d:self.copy_ids.append(d['data-copy'])
        if tag=='pre' and d.get('id','').startswith('code-'):self.pre=d['id'];self.code[self.pre]=''
        if tag=='script' and 'src' not in d:self.in_script=True
    def handle_endtag(self,tag):
        if tag=='pre':self.pre=None
        if tag=='script':self.in_script=False
    def handle_data(self,data):
        if self.pre:self.code[self.pre]+=data
        if self.in_script:self.script+=data


def main(node):
    navigation=json.loads((WORKSPACE/'notebooks/navigation.json').read_text())
    pointer=json.loads((WORKSPACE/'results/research_notebooks/current.json').read_text())
    run=Path(pointer['path'])
    paths=[WORKSPACE/'notebooks/index.html']+[WORKSPACE/'notebooks'/r['html'] for r in navigation['notebooks']]
    pages={p.resolve():Page(p.read_text(encoding='utf-8')) for p in paths}
    broken=[];links=0
    for path,page in pages.items():
        for href in page.links:
            parsed=urlsplit(href)
            if parsed.scheme or not parsed.path and not parsed.fragment:continue
            target=(path.parent/unquote(parsed.path)).resolve() if parsed.path else path
            links+=1
            if not target.exists():broken.append(dict(page=str(path),href=href,reason='missing file'))
            elif parsed.fragment and target.suffix=='.html':
                target_page=pages.get(target) or Page(target.read_text(encoding='utf-8'))
                if parsed.fragment not in target_page.ids and unquote(parsed.fragment) not in target_page.ids:
                    broken.append(dict(page=str(path),href=href,reason='missing anchor'))
    portal=pages[(WORKSPACE/'notebooks/index.html').resolve()]
    fixture=dict(cards=portal.cards,copy_ids=portal.copy_ids,code=portal.code,script=portal.script)
    fixture_path=run/'navigation_dom_fixture.json';fixture_path.write_text(json.dumps(fixture),encoding='utf-8')
    process=subprocess.run([str(node),str(WORKSPACE/'tools/check_notebook_search.js'),str(fixture_path)],capture_output=True,text=True,check=True)
    search=json.loads(process.stdout)
    result=dict(local_links_checked=links,broken_links=broken,search_script=search,
                browser_smoke_test='BLOCKED: the browser tool disallowed local file URLs; no alternate browser or server workaround was attempted',
                human_audio_listening='NOT VERIFIED: rehearse playback on the local machine')
    previous=run/'navigation_qa.json'
    if previous.is_file():(run/('navigation_qa_previous_'+uuid.uuid4().hex+'.json')).write_bytes(previous.read_bytes())
    previous.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    if broken:raise AssertionError('Broken notebook links remain visible in navigation_qa.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--node',type=Path,required=True)
    main(parser.parse_args().node)
