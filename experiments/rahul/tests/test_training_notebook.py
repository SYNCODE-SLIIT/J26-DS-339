"""Exercise the actual notebook helpers without triggering its full training cell."""
import ast
import bisect
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest

import nbformat
import torch
from torch.utils.data import Dataset

from experiments.rahul.src.inspect_transformer import DEFAULT_DATASET, mixed_length_examples
from experiments.rahul.src.model import DatasetSpec, ModelConfig, ChordRefinementTransformer, masked_cross_entropy
from experiments.rahul.src.model.context_encoder import GenreTemporalEncoder
from experiments.rahul.src.transformer_dataset import SavedSequenceDataset, collate_sequences, make_dataloader

NOTEBOOK = Path(__file__).resolve().parents[1]/'notebooks/06_train_transformer.ipynb'


class TrainingNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.notebook = nbformat.read(NOTEBOOK, as_version=4)
        cls.namespace = dict(torch=torch, Dataset=Dataset, SavedSequenceDataset=SavedSequenceDataset,
            bisect=bisect, time=time, Path=Path, json=json, hashlib=hashlib,
            masked_cross_entropy=masked_cross_entropy)
        for tag in ('column_contract','data_helpers','epoch_helper'):
            cell = next(c for c in cls.notebook.cells if tag in c.metadata.get('tags',[]))
            exec(compile(cell.source,str(NOTEBOOK), 'exec'),cls.namespace)
        cls.examples = mixed_length_examples(DEFAULT_DATASET)
        cls.spec = DatasetSpec.from_folder(DEFAULT_DATASET)

    def test_all_code_cells_parse(self):
        for cell in self.notebook.cells:
            if cell.cell_type=='code':
                ast.parse(cell.source)

    def test_exact_input_and_target_contract(self):
        expected = ('basic_chord_ids melody_pitch_ids melody_timing melody_note_mask melody_silent '
                    'key_ids mode_ids roman_ids harmonic_function_ids genre_features beat_index '
                    'bar_number beat_position event_duration relative_bar_number beat_timing_known_mask '
                    'beat_attention_mask').split()
        inputs, targets = self.namespace['select_batch'](collate_sequences(self.examples),torch.device('cpu'))
        self.assertEqual(list(inputs),expected)
        self.assertEqual(list(targets),['target_chord_ids','target_loss_mask'])
        self.assertNotIn('melody_source_ids',inputs)

    def test_json_report_is_roundtrippable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'report.json'
            self.namespace['atomic_json'](path,{'ok':True})
            self.assertEqual(json.loads(path.read_text()),{'ok':True})
            self.assertFalse(path.with_suffix('.json.tmp').exists())

    def test_cache_matches_saved_examples(self):
        cached = self.namespace['CachedSelectedDataset'](DEFAULT_DATASET/'validation_sequences.json')
        saved = SavedSequenceDataset(DEFAULT_DATASET/'validation_sequences.json')
        self.assertEqual(len(cached),len(saved))
        for index in (0,len(saved)-1):
            for key in self.namespace['INPUT_KEYS']:
                torch.testing.assert_close(cached[index]['inputs'][key],saved[index]['inputs'][key],rtol=0,atol=0)

    def test_absolute_positions_enter_temporal_projection(self):
        inputs,_ = self.namespace['select_batch'](collate_sequences(self.examples),torch.device('cpu'))
        before=GenreTemporalEncoder.temporal_values(inputs,True)
        self.assertEqual(before.shape[-1],7)
        inputs['beat_index']+=5
        inputs['bar_number']+=5
        after=GenreTemporalEncoder.temporal_values(inputs,True)
        self.assertFalse(torch.equal(before[...,5],after[...,5]))
        known=inputs['beat_timing_known_mask'][...,0]
        self.assertTrue((after[...,6][known]>before[...,6][known]).all())
        inputs['beat_timing_known_mask'].zero_()
        self.assertTrue((GenreTemporalEncoder.temporal_values(inputs,True)[...,6]==0).all())

    def test_real_epoch_backward_and_evaluation(self):
        torch.manual_seed(42)
        config=ModelConfig(d_model=32,d_ff=64,chord_dim=8,melody_dim=8,
                           harmonic_dim=8,genre_temporal_dim=8,use_absolute_timing=True)
        model=ChordRefinementTransformer(self.spec,config)
        loader=make_dataloader(self.examples,batch_size=2)
        optimizer=torch.optim.AdamW(model.parameters(),lr=0.001)
        before=model.classifier.weight.detach().clone()
        train=self.namespace['run_epoch'](model,loader,torch.device('cpu'),optimizer,log_every=0)
        self.assertEqual(train['optimizer_steps'],2)
        self.assertFalse(torch.equal(before,model.classifier.weight))
        for name,parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad,name)
            self.assertTrue(torch.isfinite(parameter.grad).all(),name)
        before=model.classifier.weight.detach().clone()
        validation=self.namespace['run_epoch'](model,loader,torch.device('cpu'),log_every=0)
        self.assertEqual(validation['optimizer_steps'],0)
        self.assertTrue(torch.equal(before,model.classifier.weight))
        self.assertEqual(validation['valid_targets'],sum(int(x['targets']['target_loss_mask'].sum()) for x in self.examples))
        self.assertEqual(validation['real_beats'],48)

    def test_per_class_metrics_include_zero_support_and_reserved_classes(self):
        helper=next(c for c in self.notebook.cells if 'class_metrics_helper' in c.metadata.get('tags',[]))
        namespace={}
        exec(compile(helper.source,str(NOTEBOOK),'exec'),namespace)
        labels=['PAD','UNK','N','C:maj','C:min','C:maj7']
        counts=torch.zeros(6,6,dtype=torch.int64)
        counts[3,3]=3
        counts[3,4]=2
        counts[4,3]=1
        counts[4,5]=2
        rows=namespace['class_prediction_rows'](counts,labels)
        self.assertEqual(len(rows),len(labels))
        self.assertEqual(rows[3]['reference_beats'],5)
        self.assertEqual(rows[3]['incorrect'],2)
        self.assertEqual(rows[3]['accuracy_recall'],3/5)
        self.assertEqual(rows[3]['precision'],3/4)
        self.assertEqual(rows[4]['accuracy_recall'],0)
        self.assertIsNone(rows[5]['accuracy_recall'])
        self.assertEqual(rows[5]['predicted_beats'],2)
        self.assertEqual(rows[5]['precision'],0)
        self.assertTrue(rows[0]['reserved'])
        self.assertFalse(rows[2]['reserved'])
        self.assertIsNone(rows[2]['precision'])
        self.assertEqual(rows[3]['most_common_predictions'][0],{'label':'C:maj','count':3})
        self.assertEqual(sum(r['reference_beats'] for r in rows),int(counts.sum()))

    def test_per_class_metrics_reject_mismatched_vocabulary(self):
        helper=next(c for c in self.notebook.cells if 'class_metrics_helper' in c.metadata.get('tags',[]))
        namespace={}
        exec(compile(helper.source,str(NOTEBOOK),'exec'),namespace)
        with self.assertRaises(ValueError):
            namespace['class_prediction_rows'](torch.zeros(2,2,dtype=torch.int64),['N'])

    def test_individual_qualities_keep_extensions_separate_and_score_exact_matches(self):
        helper=next(c for c in self.notebook.cells if 'quality_metrics_helper' in c.metadata.get('tags',[]))
        namespace={}
        exec(compile(helper.source,str(NOTEBOOK),'exec'),namespace)
        counts=torch.zeros(4,4,dtype=torch.int64)
        counts[0,0]=2
        counts[0,1]=3  # C:maj7 -> C:maj is incorrect, not collapsed to Major.
        counts[1,1]=4
        counts[3,0]=2  # D:maj7 -> C:maj7 is incorrect despite same quality.
        rows={row['reference_quality']:row for row in namespace['quality_prediction_rows'](
            counts,['C:maj7','C:maj','A:min7','D:maj7'])}
        self.assertEqual(rows['maj7']['reference_beats'],7)
        self.assertEqual(rows['maj7']['exact_chord_accuracy'],2/7)
        self.assertEqual(rows['maj']['exact_chord_accuracy'],1)
        self.assertEqual(rows['min7']['status'],'No valid test examples')
        for quality in ('6','maj9','add9'):
            self.assertIsNone(rows[quality]['exact_chord_accuracy'])
            self.assertEqual(rows[quality]['status'],'Not in saved target vocabulary')
        self.assertNotIn('N',rows)


if __name__=='__main__':
    unittest.main()
