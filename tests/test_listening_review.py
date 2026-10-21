"""Review provenance checks with fictional records, not real reviewer judgments."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_listening_review import validate_reviews

class ListeningReviewTests(unittest.TestCase):
    def test_pending_is_not_reviewed(self):
        self.assertEqual(validate_reviews({'cases':[{'key':'fixture','status':'pending'}]},['fixture'])['reviewed'],0)

    def test_review_requires_listening_and_provenance(self):
        with self.assertRaises(ValueError):
            validate_reviews({'cases':[{'key':'fixture','status':'reviewed'}]},['fixture'])

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            validate_reviews({'cases':[{'key':'fixture','status':'pending'}]*2},['fixture'])

    def test_complete_unresolved_never_authorizes_annotation(self):
        row={'key':'fixture','status':'reviewed','reviewer':'fictional test fixture','reviewed_at':'2026-10-21','evidence':'Synthetic fixture only, timestamp 1.0s','listened_original':True,'listened_references':True,'outcome':'unresolved','confidence':'low','suggested_shift_semitones':None}
        result=validate_reviews({'cases':[row]},['fixture'])
        self.assertEqual(result['reviewed_but_unresolved'],1)
        self.assertFalse(result['annotation_changes_authorized'])
