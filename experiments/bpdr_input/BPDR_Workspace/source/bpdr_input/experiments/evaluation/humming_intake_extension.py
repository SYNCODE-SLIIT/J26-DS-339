"""Prospective longer-note HumTrans development sampling, before model scoring.

Original intake is retained. Uses official-train MIDI to select long held-note
opportunities, not model predictions or success scores. No final-test audio.
"""
import io
from pathlib import Path
import uuid
import zipfile
from bpdr_input.experiments.evaluation.humming_intake import ROOT, BASE, RangeReader, session, identity
from bpdr_input.experiments.evaluation.humming_experiment import midi_notes
from bpdr_input.experiments.evaluation.transfer_study import read, write, sha


def main():
    original=read(ROOT/'intake_receipt.json');used={r['composition_id'] for r in original['records']}
    keys=read(ROOT/'train_valid_test_keys.json')['TRAIN'];out=ROOT/'extensions'/uuid.uuid4().hex;out.mkdir(parents=True)
    selected=[]
    with zipfile.ZipFile(ROOT/'all_midi.zip') as z:
        names={Path(n).stem:n for n in z.namelist() if n.lower().endswith(('.mid','.midi'))}
        for person in ['M02','M03']:
            for key in sorted(keys):
                p,song=identity(key)
                if p!=person or song in used:continue
                raw=z.read(names[key]);temp=out/'reference_probe.mid';temp.write_bytes(raw)
                notes=midi_notes(temp)
                duration=max((min(b,19.)-max(a,1.) for a,b,pitch in notes if b>1 and a<19),default=0)
                if duration<1.8:continue
                folder=out/key;folder.mkdir();midi=folder/'reference.mid';midi.write_bytes(raw)
                selected.append(dict(key=key,person_id=person,composition_id=song,split='development',official_split='train',midi_path=str(midi),midi_sha256=sha(midi),score_long_note_seconds=duration))
                used.add(song)
                if sum(r['person_id']==person for r in selected)==4:break
    write(out/'declaration.json',dict(version='prospective-score-long-note-sampling-1',original_intake_sha256=sha(ROOT/'intake_receipt.json'),
        selection=selected,rule='First four official-train keys per existing development person with a score note >=1.8 seconds within seconds1-19, globally new compositions',
        purpose='Suitability sampling for existing long stable-note BPDR protocol; not a representative humming benchmark',source_sha256=sha(__file__),
        selection_before_audio_or_model_results=True,final_test_audio_accessed=False))
    reader=RangeReader(session(),BASE+'all_wav.zip')
    with zipfile.ZipFile(reader) as z:
        names={Path(n).stem:n for n in z.namelist() if n.lower().endswith('.wav')}
        for r in selected:
            info=z.getinfo(names[r['key']])
            if info.file_size>16*1024*1024:raise ValueError('Oversized selected audio')
            path=out/r['key']/'original.wav';path.write_bytes(z.read(info));r.update(path=str(path),sha256=sha(path))
            print(r['key'],r['score_long_note_seconds'],flush=True)
    receipt=dict(original,records=original['records']+selected,extension_declaration=str(out/'declaration.json'),
        extension_bytes_transferred=reader.transferred,selection_bias='Score-selected long-note development subset, not representative; original unselected recordings retained')
    write(out/'intake_receipt.json',receipt);write(ROOT/'extension_current.json',dict(path=str(out)));print('EXTENSION',out,flush=True)


if __name__=='__main__':main()
