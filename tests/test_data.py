import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from orbit.data import (
    read_parquet,
    read_records,
    synthetic_records,
    validate_records,
    write_parquet,
)
from orbit.io import write_json, write_jsonl


class DataTests(unittest.TestCase):
    def test_synthetic_golden_contract(self) -> None:
        record = synthetic_records("train")[0]
        self.assertEqual(record["extra_info"]["query_id"], "synthetic-train-1")
        self.assertEqual(record["reward_model"], {"style": "rule", "ground_truth": "example"})
        self.assertEqual(validate_records([record], require_curriculum=True), [record])
        self.assertNotEqual(
            record["extra_info"]["query_id"], synthetic_records("val")[0]["extra_info"]["query_id"]
        )

    def test_validation_does_not_mutate_input(self) -> None:
        records = synthetic_records("train")
        result = validate_records(records)
        result[0]["extra_info"]["rubrics"][0]["points"] = 10
        self.assertEqual(records[0]["extra_info"]["rubrics"][0]["points"], 1)

    def test_bad_ids_and_duplicate_ids(self) -> None:
        records = synthetic_records("train")
        with self.assertRaises(ValueError):
            validate_records(records + records)
        for value in ("", None, 123):
            record = copy.deepcopy(records[0])
            record["extra_info"]["query_id"] = value
            with self.assertRaises(ValueError):
                validate_records([record])

    def test_nonfinite_and_nonnumeric_points(self) -> None:
        for value in (float("nan"), float("inf"), True, "1", None):
            record = synthetic_records("train")[0]
            record["extra_info"]["rubrics"][0]["points"] = value
            with self.assertRaises(ValueError):
                validate_records([record])

    def test_curriculum_requires_valid_permutation(self) -> None:
        for value in (None, [], [1], [True], [0, 0], ["0"]):
            record = synthetic_records("train")[0]
            record["extra_info"]["sorted_rubric_indices"] = value
            with self.assertRaises(ValueError):
                validate_records([record], require_curriculum=True)

    def test_missing_fields_and_prompt_key(self) -> None:
        for field in ("prompt", "data_source", "reward_model", "extra_info"):
            record = synthetic_records("train")[0]
            del record[field]
            with self.assertRaises(ValueError):
                validate_records([record])
        record = synthetic_records("train")[0]
        record["messages"] = record.pop("prompt")
        self.assertEqual(
            validate_records([record], prompt_key="messages")[0]["messages"], record["messages"]
        )

    def test_json_and_jsonl_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records = synthetic_records("val")
            for name, writer in (("data.json", write_json), ("data.jsonl", write_jsonl)):
                path = Path(directory) / name
                writer(path, records)
                self.assertEqual(read_records(path), records)

    def test_record_iterable_boundaries(self) -> None:
        for value in (None, 1, "record", b"record", synthetic_records("train")[0]):
            with self.assertRaises(ValueError):
                validate_records(value)
        self.assertEqual(
            validate_records(iter(synthetic_records("train"))), synthetic_records("train")
        )

    def test_extra_values_require_lossless_json_types(self) -> None:
        recursive: dict = {}
        recursive["self"] = recursive
        for value in (float("nan"), float("inf"), {1: "key"}, {"a": object()}, (1, 2), recursive):
            record = synthetic_records("train")[0]
            record["metadata"] = value
            with self.assertRaises(ValueError):
                validate_records([record])

    def test_strict_json_readers_reject_ambiguous_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ("json", "jsonl"):
                path = Path(directory) / f"data.{suffix}"
                for content in (
                    '{"query_id":"a","query_id":"b"}',
                    '{"points":NaN}',
                    '{"points":Infinity}',
                    '{"points":1e400}',
                ):
                    path.write_text(content if suffix == "jsonl" else "[" + content + "]")
                    with self.assertRaises(ValueError):
                        read_records(path)

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "optional pyarrow not installed")
    def test_arrow_conversion_failure_preserves_previous_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.parquet"
            baseline = synthetic_records("train")
            write_parquet(path, baseline)
            before = path.read_bytes()
            bad = synthetic_records("train") + synthetic_records("val")
            bad[0]["metadata"] = 1
            bad[1]["metadata"] = "incompatible Arrow column"
            with self.assertRaises((TypeError, ValueError)):
                write_parquet(path, bad, overwrite=True)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(directory).glob(".orbit-data-*")), [])

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "optional pyarrow not installed")
    def test_arrow_partial_write_failure_cleans_temporary(self) -> None:
        import pyarrow.parquet as parquet

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.parquet"
            write_parquet(path, synthetic_records("train"))
            before = path.read_bytes()

            def partial_write(table: object, temporary: str) -> None:
                Path(temporary).write_bytes(b"partial")
                raise OSError("injected disk failure")

            with patch.object(parquet, "write_table", side_effect=partial_write):
                with self.assertRaises(OSError):
                    write_parquet(path, synthetic_records("val"), overwrite=True)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(directory).glob(".orbit-data-*")), [])

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "optional pyarrow not installed")
    def test_parquet_roundtrip_and_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.parquet"
            records = synthetic_records("train")
            write_parquet(path, records, require_curriculum=True)
            self.assertEqual(read_parquet(path, require_curriculum=True), records)
            with self.assertRaises(FileExistsError):
                write_parquet(path, synthetic_records("val"))
            self.assertEqual(read_parquet(path), records)
            write_parquet(path, synthetic_records("val"), overwrite=True)
            self.assertEqual(read_parquet(path), synthetic_records("val"))


if __name__ == "__main__":
    unittest.main()


def test_unrepresentable_numeric_weight_is_rejected_without_overflow():
    rows = synthetic_records("train")
    rows[0]["extra_info"]["rubrics"][0]["points"] = 10**1000
    with unittest.TestCase().assertRaises(ValueError):
        validate_records(rows)
