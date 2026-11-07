import unittest
from pathlib import Path

from harness.contract import Page, Snapshot
from harness.run_eval import load_submission
from submission.solution import _passage_rate_values, build


def snapshot(snapshot_id, taken_on, pages):
    return Snapshot(
        snapshot_id=snapshot_id,
        taken_on=taken_on,
        pages=[Page(url=url, title=title, markdown=text, fetched_at=taken_on) for url, title, text in pages],
    )


class ConflictDetectionTests(unittest.TestCase):
    def test_submission_loads_through_evaluation_harness(self):
        system = load_submission(Path("submission"))
        self.assertTrue(callable(system.answer))

    def test_maps_table_values_to_their_metric_columns(self):
        values = _passage_rate_values(
            "| Term | Dividend Rate | APY |\n| 6 months | 3.45% | 3.50% |"
        )
        self.assertEqual([("3.45%", "dividend rate"), ("3.50%", "apy")], values)

    def test_detects_cross_page_stale_calculator(self):
        system = build()
        system.ingest(snapshot("current", "2026-09-25", [
            ("https://example.com/personal/savings-account", "Certificate Rates", "6-Month Certificate APY: 3.50%."),
            ("https://example.com/calculators/certificate", "Certificate Calculator", "6-Month Certificate APY: 2.95%."),
        ]))

        conflicts = system.conflicts()

        self.assertEqual(1, len(conflicts))
        self.assertEqual({"2.95%", "3.50%"}, set(conflicts[0].values))
        self.assertEqual("3.50%", conflicts[0].resolution)
        self.assertEqual(2, len(conflicts[0].sources))
        answer = system.answer("What is the 6-month certificate APY?")
        self.assertIn("3.50%", answer.text)
        self.assertNotIn("2.95%", answer.text)

    def test_detects_cross_page_conflict_without_passage_word_overlap(self):
        system = build()
        system.ingest(snapshot("current", "2026-09-25", [
            ("https://example.com/rates", "Deposit Rates", "## Certificate Rates\n\n| Term | APY |\n| 6 months | 3.50% |"),
            ("https://example.com/calculators/certificate", "CD Calculator", "A 6-mo certificate earning an annual percentage yield of 2.95 percent."),
        ]))

        conflicts = system.conflicts()

        self.assertEqual(1, len(conflicts))
        self.assertEqual({"2.95%", "3.50%"}, set(conflicts[0].values))
        self.assertEqual("3.50%", conflicts[0].resolution)

    def test_normalizes_year_and_month_terms(self):
        system = build()
        system.ingest(snapshot("current", "2026-09-25", [
            ("https://example.com/rates", "Certificate Rates", "1-year Certificate APY: 4.00%."),
            ("https://example.com/calculators/certificate", "Certificate Calculator", "12 month Certificate APY: 3.75%."),
        ]))

        conflicts = system.conflicts()

        self.assertEqual(1, len(conflicts))
        self.assertEqual("12-month", conflicts[0].topic.split()[1])

    def test_detects_change_across_versions_of_one_page(self):
        system = build()
        url = "https://example.com/personal/savings-account"
        system.ingest(snapshot("older", "2026-06-01", [(url, "Certificate Rates", "6-Month Certificate APY: 2.95%.")]))
        system.ingest(snapshot("newer", "2026-09-25", [(url, "Certificate Rates", "6-Month Certificate APY: 3.50%.")]))

        conflicts = system.conflicts()

        self.assertEqual(1, len(conflicts))
        self.assertEqual("3.50%", conflicts[0].resolution)
        self.assertIn("newest version", conflicts[0].reason)

    def test_keeps_different_terms_separate(self):
        system = build()
        system.ingest(snapshot("current", "2026-09-25", [
            ("https://example.com/personal/savings-account", "Certificate Rates", "6-Month Certificate APY: 3.50%."),
            ("https://example.com/calculators/certificate", "Certificate Calculator", "12-Month Certificate APY: 4.00%."),
        ]))

        self.assertEqual([], system.conflicts())

    def test_keeps_different_balance_tiers_separate(self):
        system = build()
        system.ingest(snapshot("current", "2026-09-25", [
            ("https://example.com/personal/savings-account", "Share Savings Rates", "Share Savings $10,000-$24,999.99 APY: 0.15%."),
            ("https://example.com/calculators/savings", "Savings Calculator", "Share Savings $25,000-$49,999.99 APY: 0.20%."),
        ]))

        self.assertEqual([], system.conflicts())


if __name__ == "__main__":
    unittest.main()
