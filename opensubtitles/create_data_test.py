"""Tests for create_data.py."""

import json
import shutil
import tempfile
import unittest
from glob import glob
from os import path

import tensorflow as tf

# from . import create_data
import create_data

_TRAIN_FILE = "\n".join([
    "matt: AAAA",  # words followed by colons are stripped.
    "[skip]",  # text in brackets is removed.
    "BBBB",
    "", "", ""  # empty lines are ignored.
    "CCCC",
    "(all laughing)",
    "c3po:",
    "- DDDD (boom!)",
    "123",  # line length will be below the test --min_length.
    "12345",  # line length will be above the test --min_length.
])

_TEST_FILE = """
aaaa
bbbb
cccc
dddd
"""


class CreateDataPipelineTest(unittest.TestCase):

    def setUp(self):
        self._temp_dir = tempfile.mkdtemp()
        self.maxDiff = None

    def tearDown(self):
        shutil.rmtree(self._temp_dir)

    def test_run(self):
        # These filenames are chosen so that their hashes will cause them to
        # be put in the train and test set respectively.
        with open(path.join(self._temp_dir, "input_train.txt"), "w") as f:
            f.write(_TRAIN_FILE.encode("utf-8").decode())
            # f.write(_TRAIN_FILE.encode("utf-8"))

        with open(path.join(self._temp_dir, "input_test.txt"), "w") as f:
            f.write(_TEST_FILE.encode("utf-8").decode())
            # f.write(_TEST_FILE.encode("utf-8"))

        create_data.run(argv=[
            "--runner=DirectRunner",
            "--sentence_files={}/*.txt".format(self._temp_dir),
            "--output_dir=" + self._temp_dir,
            "--dataset_format=TF",
            "--num_shards_test=2",
            "--num_shards_train=2",
            "--min_length=4",
            "--max_length=5",
            "--train_split=0.5",
        ])

        self.assertCountEqual(
            [path.join(self._temp_dir, expected_file) for expected_file in
             ["train-00000-of-00002.tfrecord",
              "train-00001-of-00002.tfrecord"]],
            sorted(glob(path.join(self._temp_dir, "train-*")))
        )
        self.assertCountEqual(
            [path.join(self._temp_dir, expected_file) for expected_file in
             ["test-00000-of-00002.tfrecord",
              "test-00001-of-00002.tfrecord"]],
            sorted(glob(path.join(self._temp_dir, "test-*")))
        )

        train_examples = self._read_examples("train-*")
        expected_train_examples = [
            self.create_example(
                ["AAAA"], "BBBB", "input_train.txt"),
            self.create_example(
                ["AAAA", "BBBB"], "CCCC", "input_train.txt"),
            self.create_example(
                ["AAAA", "BBBB", "CCCC"], "DDDD", "input_train.txt"),
        ]
        self.assertCountEqual(
            expected_train_examples,
            train_examples
        )

        test_examples = self._read_examples("test-*")
        expected_test_examples = [
            self.create_example(
                ["aaaa"], "bbbb", "input_test.txt"),
            self.create_example(
                ["aaaa", "bbbb"], "cccc", "input_test.txt"),
            self.create_example(
                ["aaaa", "bbbb", "cccc"], "dddd", "input_test.txt"),
        ]
        self.assertCountEqual(
            expected_test_examples,
            test_examples
        )

    def create_example(self, previous_lines, line, file_id):
        features = create_data.create_example(previous_lines, line, file_id)
        example = tf.train.Example()
        for feature_name, feature_value in features.items():
            example.features.feature[feature_name].bytes_list.value.append(feature_value.encode("utf-8"))
        return example

    def _read_examples(self, pattern):
        examples = []
        for file_name in sorted(glob(path.join(self._temp_dir, pattern))):
            for record in tf.data.TFRecordDataset(file_name):
                example = tf.train.Example()
                example.ParseFromString(record.numpy())
                examples.append(example)
        return examples

    def test_run_json(self):
        # These filenames are chosen so that their hashes will cause them to
        # be put in the train and test set respectively.
        with open(path.join(self._temp_dir, "input_train.txt"), "w") as f:
            f.write(_TRAIN_FILE.encode("utf-8").decode())

        with open(path.join(self._temp_dir, "input_test.txt"), "w") as f:
            f.write(_TEST_FILE.encode("utf-8").decode())

        create_data.run(argv=[
            "--runner=DirectRunner",
            "--sentence_files={}/*.txt".format(self._temp_dir),
            "--output_dir=" + self._temp_dir,
            "--dataset_format=JSON",
            "--num_shards_test=2",
            "--num_shards_train=2",
            "--min_length=4",
            "--max_length=5",
            "--train_split=0.5",
        ])

        self.assertCountEqual(
            [path.join(self._temp_dir, expected_file) for expected_file in
             ["train-00000-of-00002.json",
              "train-00001-of-00002.json"]],
            sorted(glob(path.join(self._temp_dir, "train-*")))
        )
        self.assertCountEqual(
            [path.join(self._temp_dir, expected_file) for expected_file in
             ["test-00000-of-00002.json",
              "test-00001-of-00002.json"]],
            sorted(glob(path.join(self._temp_dir, "test-*")))
        )

        train_examples = self._read_json_examples("train-*")
        expected_train_examples = [
            create_data.create_example(
                ["AAAA"], "BBBB", "input_train.txt"),
            create_data.create_example(
                ["AAAA", "BBBB"], "CCCC", "input_train.txt"),
            create_data.create_example(
                ["AAAA", "BBBB", "CCCC"], "DDDD", "input_train.txt"),
        ]
        self.assertCountEqual(
            expected_train_examples,
            train_examples
        )

        test_examples = self._read_json_examples("test-*")
        expected_test_examples = [
            create_data.create_example(
                ["aaaa"], "bbbb", "input_test.txt"),
            create_data.create_example(
                ["aaaa", "bbbb"], "cccc", "input_test.txt"),
            create_data.create_example(
                ["aaaa", "bbbb", "cccc"], "dddd", "input_test.txt"),
        ]
        self.assertCountEqual(
            expected_test_examples,
            test_examples
        )

    def _read_json_examples(self, pattern):
        examples = []
        for file_name in sorted(glob(path.join(self._temp_dir, pattern))):
            with open(file_name) as f:
                for line in f:
                    examples.append(json.loads(line))
        return examples


if __name__ == "__main__":
    unittest.main()
