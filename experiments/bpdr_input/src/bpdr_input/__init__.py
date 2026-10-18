"""BPDR workspace with grouped source files and stable public imports.

Physical folders express responsibility. Existing bpdr_input.model, pipeline,
training and other module imports remain valid for saved code and checkpoints.
"""
from pathlib import Path
import importlib
import importlib.abc
import importlib.util
import importlib.machinery
import json
import sys

__version__ = "0.1.0"
SOURCE_ROOT = Path(__file__).resolve().parent
COMPONENT_ROOT = next(p for p in SOURCE_ROOT.parents if p.name == 'bpdr_input' and (p/'pyproject.toml').is_file())
_SOURCE_MAP = json.loads((SOURCE_ROOT/'_module_map.json').read_text(encoding='utf-8'))
_ALIASES = {'analysis': 'audio.analysis', 'anchor_feasibility': 'experiments.audio.anchor_feasibility', 'app': 'application.app', 'audit_diagnostics': 'experiments.audio.audit_diagnostics', 'boundary_comparison': 'experiments.audio.boundary_comparison', 'boundary_renderer': 'experiments.audio.boundary_renderer', 'bpdr_showcase': 'reporting.bpdr_showcase', 'comparison_preparation': 'experiments.learning.comparison_preparation', 'completion_release': 'packaging.review_package', 'context_ablation': 'experiments.audio.context_ablation', 'context_ablation_experiment': 'experiments.audio.context_ablation_experiment', 'context_ablation_scoring': 'experiments.audio.context_ablation_scoring', 'contracts': 'audio.contracts', 'controlled_study': 'experiments.learning.controlled_study', 'controller_fixtures': 'experiments.audio.controller_fixtures', 'coverage': 'preparation.diagnostics.coverage', 'cue_features': 'experiments.evaluation.cue_features', 'data_builder': 'preparation.sources.data_builder', 'data_sources': 'preparation.sources.data_sources', 'dataset': 'preparation.sources.dataset', 'demo': 'experiments.audio.demo', 'development_boundary_observability': 'preparation.diagnostics.development_boundary_observability', 'development_preparation': 'preparation.pairs.development_preparation', 'environmental_noise_study': 'experiments.evaluation.environmental_noise_study', 'estimated_harmonic_experiment': 'experiments.audio.estimated_harmonic_experiment', 'estimated_harmonic_renderer': 'experiments.audio.estimated_harmonic_renderer', 'examples': 'modeling.examples', 'expanded_comparison': 'experiments.learning.expanded_comparison', 'expanded_experiment_preparation': 'experiments.learning.expanded_experiment_preparation', 'feature_weighting_control': 'experiments.learning.feature_weighting_control', 'fine_pitch': 'experiments.learning.fine_pitch', 'fine_pitch_bpdr_check': 'experiments.learning.fine_pitch_bpdr_check', 'fine_pitch_learning_curve': 'experiments.learning.fine_pitch_learning_curve', 'fine_pitch_mse_check': 'experiments.learning.fine_pitch_mse_check', 'fine_pitch_overfit': 'experiments.learning.fine_pitch_overfit', 'fine_pitch_overfit_v2': 'experiments.learning.fine_pitch_overfit_v2', 'fine_pitch_relative_check': 'experiments.learning.fine_pitch_relative_check', 'framewise_audit': 'repair.framewise_audit', 'framewise_diagnostics': 'experiments.audio.framewise_diagnostics', 'framewise_fixtures': 'experiments.audio.framewise_fixtures', 'framewise_renderer': 'repair.framewise_renderer', 'frozen_development_evaluation': 'experiments.learning.frozen_development_evaluation', 'generalization_probe': 'experiments.evaluation.generalization_probe', 'harmonic_separation': 'experiments.audio.harmonic_separation', 'harmonic_separation_experiment': 'experiments.audio.harmonic_separation_experiment', 'humming_experiment': 'experiments.evaluation.humming_experiment', 'humming_intake': 'experiments.evaluation.humming_intake', 'humming_intake_extension': 'experiments.evaluation.humming_intake_extension', 'humming_learning': 'experiments.learning.humming_learning', 'input_response_diagnosis': 'experiments.learning.input_response_diagnosis', 'interventions': 'repair.interventions', 'inventory': 'preparation.sources.inventory', 'jamendo_batch': 'preparation.sources.jamendo_batch', 'jamendo_yield': 'preparation.sources.jamendo_yield', 'known_control_diagnostics': 'experiments.audio.known_control_diagnostics', 'known_source_control': 'experiments.audio.known_source_control', 'laboratory': 'preparation.pairs.laboratory', 'learning_control': 'experiments.learning.learning_control', 'losses': 'modeling.losses', 'matched_comparison': 'experiments.learning.matched_comparison', 'metrics': 'modeling.metrics', 'model': 'modeling.model', 'model_pins': 'audio.model_pins', 'model_runtime': 'repair.model_runtime', 'multisource_control': 'experiments.learning.multisource_control', 'noise_policy_v2': 'repair.noise_policy_v2', 'noise_preview': 'repair.noise_preview', 'noise_study': 'experiments.evaluation.noise_study', 'notebook_evidence': 'reporting.notebook_evidence', 'notebooks': 'reporting.notebooks', 'phase_comparison': 'experiments.audio.phase_comparison', 'phase_feasibility': 'experiments.audio.phase_feasibility', 'phase_regression_diagnostics': 'experiments.audio.phase_regression_diagnostics', 'phase_renderer': 'experiments.audio.phase_renderer', 'pipeline': 'audio.pipeline', 'plateau_comparison': 'experiments.learning.plateau_comparison', 'plateau_objective': 'experiments.learning.plateau_objective', 'prediction_adapter': 'repair.prediction_adapter', 'prediction_preview': 'experiments.audio.prediction_preview', 'preparation_report': 'reporting.preparation_report', 'presentation_app': 'application.legacy_interface', 'presentation_baselines': 'experiments.evaluation.baseline_evaluation', 'presentation_release': 'packaging.release_builder', 'presentation_statistics': 'reporting.paired_statistics', 'presentation_study': 'experiments.evaluation.transfer_study', 'real_input_diagnosis': 'experiments.learning.real_input_diagnosis', 'real_learning_v2': 'experiments.learning.real_learning_v2', 'reference_cue_ceiling': 'experiments.evaluation.reference_cue_ceiling', 'renderer_calibration': 'experiments.audio.renderer_calibration', 'renderer_candidates': 'experiments.audio.renderer_candidates', 'repair_controller': 'repair.repair_controller', 'review_bundle': 'preparation.diagnostics.review_bundle', 'rule_correction': 'repair.rule_correction', 'rule_evaluation': 'experiments.evaluation.rule_evaluation', 'runtime_cue_diagnosis': 'experiments.evaluation.runtime_cue_diagnosis', 'saved_results': 'application.saved_results', 'segment_phase': 'experiments.audio.segment_phase', 'segment_phase_experiment': 'experiments.audio.segment_phase_experiment', 'segment_phase_scoring': 'experiments.audio.segment_phase_scoring', 'separation_adapter': 'preparation.sources.separation_adapter', 'shared_batch_intake': 'preparation.sources.shared_batch_intake', 'shared_export_audit': 'preparation.sources.shared_export_audit', 'shared_exports': 'preparation.sources.shared_exports', 'shared_pair_audit': 'preparation.pairs.shared_pair_audit', 'shared_screen_diagnostics': 'preparation.diagnostics.shared_screen_diagnostics', 'source_estimation_experiment': 'experiments.audio.source_estimation_experiment', 'source_estimator': 'experiments.audio.source_estimator', 'source_transfer': 'experiments.audio.source_transfer', 'support_acoustic_pilot': 'preparation.diagnostics.support_acoustic_pilot', 'support_development_audit': 'preparation.diagnostics.support_development_audit', 'support_development_packaging': 'preparation.pairs.support_development_packaging', 'support_fragmentation': 'preparation.diagnostics.support_fragmentation', 'support_geometry': 'preparation.diagnostics.support_geometry', 'training': 'modeling.training', 'training_reference_completion': 'preparation.pairs.training_reference_completion', 'training_reference_completion_v2': 'preparation.pairs.training_reference_completion_v2', 'training_reference_preparation': 'preparation.pairs.training_reference_preparation', 'training_support_audit': 'preparation.diagnostics.training_support_audit', 'training_support_packaging': 'preparation.pairs.training_support_packaging', 'verify_dataset': 'preparation.sources.verify_dataset', 'legacy_interface': 'application.legacy_interface', 'transfer_study': 'experiments.evaluation.transfer_study', 'baseline_evaluation': 'experiments.evaluation.baseline_evaluation', 'paired_statistics': 'reporting.paired_statistics', 'release_builder': 'packaging.release_builder', 'review_package': 'packaging.review_package'}


def module_source(filename):
    """Find one maintained source regardless of its responsibility folder."""
    name = Path(filename).name
    if name == '__init__.py':return SOURCE_ROOT/name
    stem = Path(name).stem
    return SOURCE_ROOT/_SOURCE_MAP.get(stem,name)


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, fullname, canonical, path):
        self.fullname,self.canonical,self.path=fullname,canonical,str(path)

    def create_module(self, spec):
        return importlib.import_module(self.canonical)

    def exec_module(self, module):
        pass  # Canonical module was initialized by import_module.

    def get_code(self, fullname):
        return importlib.machinery.SourceFileLoader(fullname,self.path).get_code(fullname)


class _AliasFinder(importlib.abc.MetaPathFinder):
    _bpdr_alias_finder=True

    def find_spec(self, fullname, path=None, target=None):
        prefix=__name__+'.'
        if fullname.startswith(prefix) and fullname[len(prefix):] in _ALIASES:
            old=fullname[len(prefix):];canonical=prefix+_ALIASES[old]
            return importlib.util.spec_from_loader(fullname,_AliasLoader(fullname,canonical,module_source(old+'.py')))
        return None


if not any(getattr(f,'_bpdr_alias_finder',False) for f in sys.meta_path):
    sys.meta_path.append(_AliasFinder())
