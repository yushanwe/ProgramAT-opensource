import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import copilot_db
import implementation_traceability


class ImplementationTraceabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / 'traceability.db')
        self.db_patch = patch.object(copilot_db, 'DB_PATH', self.db_path)
        self.db_patch.start()
        copilot_db.init_database()
        implementation_traceability.init_traceability_table()

    def tearDown(self):
        self.db_patch.stop()
        self.tmpdir.cleanup()

    def test_upsert_and_get_traceability_record(self):
        created = implementation_traceability.upsert_traceability_record(
            mode='create',
            issue_number=263,
            tool_name='Uber Car Identifier',
            implementation_summary={'model': 'gemini/gemini-3.1-flash-lite'},
            metadata={'source': 'unit_test'},
        )
        self.assertEqual(created['issue_number'], 263)
        self.assertEqual(created['tool_path'], 'tools/uber_car_identifier.py')
        self.assertEqual(created['implementation_summary']['model'], 'gemini/gemini-3.1-flash-lite')

        updated = implementation_traceability.upsert_traceability_record(
            mode='create',
            issue_number=263,
            pr_number=300,
            commit_sha='abc123',
            workflow_run_id=99,
        )
        self.assertEqual(updated['pr_number'], 300)
        self.assertEqual(updated['workflow_run_id'], 99)
        fetched = implementation_traceability.get_traceability_record(issue_number=263)
        self.assertEqual(fetched['commit_sha'], 'abc123')

    def test_extract_implementation_summary(self):
        summary = implementation_traceability.extract_implementation_summary(
            """<!-- USER_SUMMARY_START -->
100% Complete
<!-- USER_SUMMARY_END -->

<!-- EXPERT_DETAIL_START -->
### Implementation summary

- Model: `gemini/gemini-3.1-flash-lite`
- Strategy: single-frame VLM
- Execution mode: streaming
- Runtime input: search_criteria
- Validation: passed
<!-- EXPERT_DETAIL_END -->
"""
        )
        self.assertEqual(summary['model'], 'gemini/gemini-3.1-flash-lite')
        self.assertEqual(summary['strategy'], 'single-frame VLM')
        self.assertEqual(summary['execution_mode'], 'streaming')
        self.assertEqual(summary['runtime_input'], 'search_criteria')
        self.assertEqual(summary['validation'], 'passed')

    def test_sanitize_action_logs_preserves_signal_and_removes_noise(self):
        raw_logs = """
Collecting package metadata
Requirement already satisfied: pygithub
Progress
### Current analysis
Validation failed in backend/validate_generated_tools.py
gh api repos/example
Authorization: Bearer secret-token
Committed changes to tools/uber_car_identifier.py
"""
        sanitized = implementation_traceability.sanitize_action_logs(raw_logs, max_chars=5000)
        self.assertIn('Validation failed', sanitized)
        self.assertIn('Committed changes', sanitized)
        self.assertNotIn('Requirement already satisfied', sanitized)
        self.assertNotIn('gh api repos/example', sanitized)
        self.assertNotIn('secret-token', sanitized)


if __name__ == '__main__':
    unittest.main()
