import unittest
from ir_datasets_longeval import load

class TestOfficialDatasets(unittest.TestCase):
    def test_sci_dataset(self):
        dataset = load("longeval-sci/2024-11/train")

        expected_queries = {"ce5bfacf-8652-4bc1-a5b0-6144a917fb1c": "streptomyces"}

        self.assertIsNotNone(dataset)
        example_doc = dataset.docs_iter().__next__()

        self.assertEqual("11699524", example_doc.doc_id)
        actual_queries = {i.query_id: i.default_text() for i in dataset.queries_iter()}
        self.assertEqual(393, len(actual_queries))
        for k, v in expected_queries.items():
            self.assertEqual(v, actual_queries[k])

        self.assertEqual(4262, len(list(dataset.qrels_iter())))
        self.assertEqual(2024, dataset.get_timestamp().year)
        self.assertEqual([], dataset.get_past_datasets())
        docs_store = dataset.docs_store()
        self.assertEqual("11699524", docs_store.get("11699524").doc_id)
        self.assertIn("level of customers", docs_store.get("11699524").default_text())

    def test_sci_default_text_for_document_without_abstract(self):
        dataset = load("longeval-sci/2024-11/train")

        docs_store = dataset.docs_store()
        self.assertEqual("Migration, Social Norms and Experiences of Female Genital Mutilation (FGM) among Somali Communities in the UK", docs_store.get("154684342").default_text())