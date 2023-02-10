"""Tests for create_data_v2.py."""

import copy
import json
import shutil
import tempfile
import unittest
from glob import glob
from os import path

import tensorflow as tf

import create_data_v2


class CreateDataPipelineTest(unittest.TestCase):
    """Test running the pipeline end-to-end."""

    def setUp(self):
        self._temp_dir = tempfile.mkdtemp()
        self.maxDiff = None

    def tearDown(self):
        shutil.rmtree(self._temp_dir)

    def test_run(self):
        with open("reddit/testdata/simple_thread.json") as f:
            comments = json.loads(f.read())

        # Duplicate the thread with a different ID, chosing a link_id that
        # will be put in the test set.
        test_comments = []
        for comment in comments:
            test_comment = copy.copy(comment)
            # first comments under the submission
            # need to set their parent to thread id
            if test_comment['link_id'] == test_comment['parent_id']:
                test_comment['parent_id'] = "t4_testthread2"     
            test_comment['link_id'] = "t4_testthread2"
            test_comments.append(test_comment)

        create_data_v2.run(argv=[
            "--runner=DirectRunner",
            "--reddit_table=ignored",
            "--output_dir=" + self._temp_dir,
            "--dataset_format=TF",
            "--num_shards=2",
        ],
                           comments=(comments + test_comments))

        self.assertCountEqual([
            path.join(self._temp_dir, expected_file) for expected_file in
            ["all-00000-of-00002.tfrecord", "all-00001-of-00002.tfrecord"]
        ], glob(path.join(self._temp_dir, "all-*")))

        train_examples = self._read_examples("all-*")
        expected_train_examples_1 = [
            self._create_example({
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>CCCC",
            }),
            self._create_example({
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>DDDD<sep>EEEE",
            }),
            self._create_example({
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>DDDD<sep>too long to create an example",
            }),      
            self._create_example({
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>DDDD<sep>123",
            }),
            self._create_example({
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>FFFF",
            })]

        expected_train_examples_2 = [
            self._create_example({
                'subreddit': "subreddit-A",
                'thread_id': "testthread2",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>CCCC",
            }),
            self._create_example({
                'subreddit': "subreddit-A",
                'thread_id': "testthread2",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>DDDD<sep>EEEE",
            }),
            self._create_example({
                'subreddit': "subreddit-A",
                'thread_id': "testthread2",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>DDDD<sep>too long to create an example",
            }),      
            self._create_example({
                'subreddit': "subreddit-A",
                'thread_id': "testthread2",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>DDDD<sep>123",
            }),
            self._create_example({
                'subreddit': "subreddit-A",
                'thread_id': "testthread2",
                'comments': "[submission]<sep>FFFF",
            })]
        self.assertCountEqual(expected_train_examples_1+expected_train_examples_2, train_examples)


    def test_run_json(self):
        with open("reddit/testdata/simple_thread.json") as f:
            comments = json.loads(f.read())

        # Duplicate the thread with a different ID, chosing a link_id that
        # will be put in the test set.
        test_comments = []
        for comment in comments:
            test_comment = copy.copy(comment)
            # first comments under the submission
            # need to set their parent to thread id
            if test_comment['link_id'] == test_comment['parent_id']:
                test_comment['parent_id'] = "t4_testthread2"     
            test_comment['link_id'] = "t4_testthread2"
            test_comments.append(test_comment)

        create_data_v2.run(argv=[
            "--runner=DirectRunner",
            "--reddit_table=ignored",
            "--output_dir=" + self._temp_dir,
            "--dataset_format=JSON",
            "--num_shards=2",
        ], comments=(comments + test_comments))

        self.assertCountEqual([
            path.join(self._temp_dir, expected_file) for expected_file in
            ["all-00000-of-00002.json", "all-00001-of-00002.json"]
        ], glob(path.join(self._temp_dir, "all-*")))

        train_examples = self._read_json_examples("all-*")
        expected_train_examples_1 = [
            {
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>CCCC",
            },
            {
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>DDDD<sep>EEEE",
            },
            {
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>DDDD<sep>too long to create an example",
            },      
            {
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>DDDD<sep>123",
            },
           {
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>FFFF",
            }]

        expected_train_examples_2 = [
            {
                'subreddit': "subreddit-A",
                'thread_id': "testthread2",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>CCCC",
            },
            {
                'subreddit': "subreddit-A",
                'thread_id': "testthread2",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>DDDD<sep>EEEE",
            },
            {
                'subreddit': "subreddit-A",
                'thread_id': "testthread2",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>DDDD<sep>too long to create an example",
            },      
            {
                'subreddit': "subreddit-A",
                'thread_id': "testthread2",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>DDDD<sep>123",
            },
            {
                'subreddit': "subreddit-A",
                'thread_id': "testthread2",
                'comments': "[submission]<sep>FFFF",
            }]
        self.assertCountEqual(expected_train_examples_1+expected_train_examples_2, train_examples)


    def test_run_multilingual(self):
        with open("reddit/testdata/simple_thread.json") as f:
            comments = json.loads(f.read())

        # Duplicate the thread with a different ID, chosing a link_id that
        # will be put in the test set.
        test_comments = []
        for comment in comments:
            test_comment = copy.copy(comment)
            # first comments under the submission
            # need to set their parent to thread id
            if test_comment['link_id'] == test_comment['parent_id']:
                test_comment['parent_id'] = "t4_testthread2"     
            test_comment['link_id'] = "t4_testthread2"
            test_comment['body'] = "Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!"
            test_comments.append(test_comment)

        create_data_v2.run(argv=[
            "--runner=DirectRunner",
            "--reddit_table=ignored",
            "--output_dir=" + self._temp_dir,
            "--detect_lang=true",
            "--dataset_format=TF",
            "--num_shards=2",
        ], comments=(comments + test_comments))

        self.assertCountEqual([
            path.join(self._temp_dir, expected_file) for expected_file in
            ["en-00000-of-00002.tfrecord", "en-00001-of-00002.tfrecord", 
             "tr-00000-of-00002.tfrecord", "tr-00001-of-00002.tfrecord",
             "other-00000-of-00002.tfrecord", "other-00001-of-00002.tfrecord"]
        ], glob(path.join(self._temp_dir, "*")))



        train_examples = self._read_examples("*")
        expected_train_examples_1 = [
            self._create_example({
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>CCCC",
                'language':'en'
            }),
            self._create_example({
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>DDDD<sep>EEEE",
                'language':'so'
            }),
            self._create_example({
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>DDDD<sep>too long to create an example",
                'language':'en'
            }),      
            self._create_example({
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>DDDD<sep>123",
                'language':'en'
            }),
            self._create_example({
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>FFFF",
                'language':'en'
            })]

        expected_train_examples_2 = [
            self._create_example({
                'subreddit': "subreddit-A",
                'thread_id': "testthread2",
                'comments': "[submission]<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!",
                'language':'tr',
            }),
            self._create_example({
                'subreddit': "subreddit-A",
                'thread_id': "testthread2",
                'comments': "[submission]<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!",
                'language':'tr'
            }),
            self._create_example({
                'subreddit': "subreddit-A",
                'thread_id': "testthread2",
                'comments': "[submission]<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!",
                'language':'tr'
            }),      
            self._create_example({
                'subreddit': "subreddit-A",
                'thread_id': "testthread2",
                'comments': "[submission]<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!",
                'language':'tr'
            }),
            self._create_example({
                'subreddit': "subreddit-A",
                'thread_id': "testthread2",
                'comments': "[submission]<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!",
                'language':'tr'
            })]
        self.assertCountEqual(expected_train_examples_1+expected_train_examples_2, train_examples)


    def test_run_multilingual_json(self):
        with open("reddit/testdata/simple_thread.json") as f:
            comments = json.loads(f.read())

        # Duplicate the thread with a different ID, chosing a link_id that
        # will be put in the test set.
        test_comments = []
        for comment in comments:
            test_comment = copy.copy(comment)
            # first comments under the submission
            # need to set their parent to thread id
            if test_comment['link_id'] == test_comment['parent_id']:
                test_comment['parent_id'] = "t4_testthread2"     
            test_comment['link_id'] = "t4_testthread2"
            test_comment['body'] = "Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!"
            test_comments.append(test_comment)

        create_data_v2.run(argv=[
            "--runner=DirectRunner",
            "--reddit_table=ignored",
            "--output_dir=" + self._temp_dir,
            "--detect_lang=true",
            "--dataset_format=JSON",
            "--num_shards=2",
        ], comments=(comments + test_comments))

        self.assertCountEqual([
            path.join(self._temp_dir, expected_file) for expected_file in
            ["en-00000-of-00002.json", "en-00001-of-00002.json", 
             "tr-00000-of-00002.json", "tr-00001-of-00002.json",
             "other-00000-of-00002.json", "other-00001-of-00002.json"]
        ], glob(path.join(self._temp_dir, "*")))



        train_examples = self._read_json_examples("*")
        expected_train_examples_1 = [
            {
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>CCCC",
                'language':'en'
            },
            {
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>DDDD<sep>EEEE",
                'language':'so'
            },
            {
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>DDDD<sep>too long to create an example",
                'language':'en'
            },      
            {
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>DDDD<sep>123",
                'language':'en'
            },
            {
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>FFFF",
                'language':'en'
            }]

        expected_train_examples_2 = [
            {
                'subreddit': "subreddit-A",
                'thread_id': "testthread2",
                'comments': "[submission]<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!",
                'language':'tr',
            },
            {
                'subreddit': "subreddit-A",
                'thread_id': "testthread2",
                'comments': "[submission]<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!",
                'language':'tr'
            },
            {
                'subreddit': "subreddit-A",
                'thread_id': "testthread2",
                'comments': "[submission]<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!",
                'language':'tr'
            },      
            {
                'subreddit': "subreddit-A",
                'thread_id': "testthread2",
                'comments': "[submission]<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!",
                'language':'tr'
            },
            {
                'subreddit': "subreddit-A",
                'thread_id': "testthread2",
                'comments': "[submission]<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!",
                'language':'tr'
            }]
        self.assertCountEqual(expected_train_examples_1+expected_train_examples_2, train_examples)


    def test_run_multilingual_no_single(self):
        with open("reddit/testdata/simple_thread.json") as f:
            comments = json.loads(f.read())

        # Duplicate the thread with a different ID, chosing a link_id that
        # will be put in the test set.
        test_comments = []
        for comment in comments:
            test_comment = copy.copy(comment)
            # first comments under the submission
            # need to set their parent to thread id
            if test_comment['link_id'] == test_comment['parent_id']:
                test_comment['parent_id'] = "t4_testthread2"     
            test_comment['link_id'] = "t4_testthread2"
            test_comment['body'] = "Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!"
            test_comments.append(test_comment)

        create_data_v2.run(argv=[
            "--runner=DirectRunner",
            "--reddit_table=ignored",
            "--output_dir=" + self._temp_dir,
            "--detect_lang=true",
            "--skip_single_comment=true",
            "--dataset_format=TF",
            "--num_shards=2",
        ], comments=(comments + test_comments))

        self.assertCountEqual([
            path.join(self._temp_dir, expected_file) for expected_file in
            ["en-00000-of-00002.tfrecord", "en-00001-of-00002.tfrecord", 
             "tr-00000-of-00002.tfrecord", "tr-00001-of-00002.tfrecord",
             "other-00000-of-00002.tfrecord", "other-00001-of-00002.tfrecord"]
        ], glob(path.join(self._temp_dir, "*")))



        train_examples = self._read_examples("*")
        expected_train_examples_1 = [
            self._create_example({
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>CCCC",
                'language':'en'
            }),
            self._create_example({
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>DDDD<sep>EEEE",
                'language':'so'
            }),
            self._create_example({
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>DDDD<sep>too long to create an example",
                'language':'en'
            }),      
            self._create_example({
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>DDDD<sep>123",
                'language':'en'
            })]

        expected_train_examples_2 = [
            self._create_example({
                'subreddit': "subreddit-A",
                'thread_id': "testthread2",
                'comments': "[submission]<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!",
                'language':'tr',
            }),
            self._create_example({
                'subreddit': "subreddit-A",
                'thread_id': "testthread2",
                'comments': "[submission]<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!",
                'language':'tr'
            }),
            self._create_example({
                'subreddit': "subreddit-A",
                'thread_id': "testthread2",
                'comments': "[submission]<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!",
                'language':'tr'
            }),      
            self._create_example({
                'subreddit': "subreddit-A",
                'thread_id': "testthread2",
                'comments': "[submission]<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!",
                'language':'tr'
            })]
        self.assertCountEqual(expected_train_examples_1+expected_train_examples_2, train_examples)


    def test_run_multilingual_json_no_single(self):
        with open("reddit/testdata/simple_thread.json") as f:
            comments = json.loads(f.read())

        # Duplicate the thread with a different ID, chosing a link_id that
        # will be put in the test set.
        test_comments = []
        for comment in comments:
            test_comment = copy.copy(comment)
            # first comments under the submission
            # need to set their parent to thread id
            if test_comment['link_id'] == test_comment['parent_id']:
                test_comment['parent_id'] = "t4_testthread2"     
            test_comment['link_id'] = "t4_testthread2"
            test_comment['body'] = "Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!"
            test_comments.append(test_comment)

        create_data_v2.run(argv=[
            "--runner=DirectRunner",
            "--reddit_table=ignored",
            "--output_dir=" + self._temp_dir,
            "--detect_lang=true",
            "--skip_single_comment=true",
            "--dataset_format=JSON",
            "--num_shards=2",
        ], comments=(comments + test_comments))

        self.assertCountEqual([
            path.join(self._temp_dir, expected_file) for expected_file in
            ["en-00000-of-00002.json", "en-00001-of-00002.json", 
             "tr-00000-of-00002.json", "tr-00001-of-00002.json",
             "other-00000-of-00002.json", "other-00001-of-00002.json"]
        ], glob(path.join(self._temp_dir, "*")))



        train_examples = self._read_json_examples("*")
        expected_train_examples_1 = [
            {
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>CCCC",
                'language':'en'
            },
            {
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>DDDD<sep>EEEE",
                'language':'so'
            },
            {
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>DDDD<sep>too long to create an example",
                'language':'en'
            },      
            {
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>DDDD<sep>123",
                'language':'en'
            }]

        expected_train_examples_2 = [
            {
                'subreddit': "subreddit-A",
                'thread_id': "testthread2",
                'comments': "[submission]<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!",
                'language':'tr',
            },
            {
                'subreddit': "subreddit-A",
                'thread_id': "testthread2",
                'comments': "[submission]<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!",
                'language':'tr'
            },
            {
                'subreddit': "subreddit-A",
                'thread_id': "testthread2",
                'comments': "[submission]<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!",
                'language':'tr'
            },      
            {
                'subreddit': "subreddit-A",
                'thread_id': "testthread2",
                'comments': "[submission]<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!<sep>Burada Turkce bir yazi yaziyor ve dil modeli bunu anlamalidir!",
                'language':'tr'
            }]
        self.assertCountEqual(expected_train_examples_1+expected_train_examples_2, train_examples)


    def _read_examples(self, pattern):
        examples = []
        for file_name in sorted(glob(path.join(self._temp_dir, pattern))):
            for record in tf.data.TFRecordDataset(file_name):
                example = tf.train.Example()
                example.ParseFromString(record.numpy())
                examples.append(example)
        return examples

    def _read_json_examples(self, pattern):
        examples = []
        for file_name in glob(path.join(self._temp_dir, pattern)):
            with open(file_name) as f:
                for line in f:
                    examples.append(json.loads(line))
        return examples

    @staticmethod
    def _create_example(features):
        example = tf.train.Example()
        for feature_name, feature_value in features.items():
            example.features.feature[feature_name].bytes_list.value.append(
                feature_value.encode("utf-8"))
        return example


class CreateDataTest(unittest.TestCase):
    """Test individual helper functions."""
    
    maxDiff = None

    def test_normalise_comment(self):
        comment = create_data_v2.normalise_comment({
            'body':
            "ABC EFG HIJ KLM NOP",
            'score_hidden':
            None,
            'archived':
            None,
            'name':
            None,
            'author_flair_text':
            None,
            'downs':
            None,
            'created_utc':
            "1520704245",
            'subreddit_id':
            "t5_AAAAA",
            'link_id':
            "t3_BBBBB",
            'parent_id':
            "t1_CCCCC",
            'score':
            "1",
            'retrieved_on':
            "1525020075",
            'controversiality':
            "0",
            'gilded':
            "0",
            'id':
            "DDDDD",
            'subreddit':
            "EEEEE",
            'author':
            "FFFFF",
            'ups':
            None,
            'distinguished':
            None,
            'author_flair_css_class':
            None,
        })
        self.assertEqual(
            comment,
            dict(
                body="ABC EFG HIJ KLM NOP",
                thread_id="BBBBB",
                parent_id="CCCCC",
                id="DDDDD",
                subreddit="EEEEE",
                author="FFFFF",
            ))

    def test_generate_paths(self):
        with open("reddit/testdata/simple_thread.json") as f:
            comments = json.loads(f.read())
        comments = [
            create_data_v2.normalise_comment(comment) for comment in comments
        ]

        # default
        paths = list(create_data_v2.generate_paths_for_thread(comments))
        self.assertCountEqual(
            [['testthread', 'id-A', 'id-B', 'id-C'],
            ['testthread', 'id-A', 'id-B', 'id-D', 'id-E'],
            ['testthread', 'id-A', 'id-B', 'id-D', 'id-too-long'],
            ['testthread', 'id-A', 'id-B', 'id-D', 'id-too-short'],
            ['testthread', 'id-no-replies']], paths)

    @staticmethod
    def _create_test_comment(id, parent_id):
        return dict(
            body="body",
            thread_id="thread_id",
            parent_id=parent_id,
            id=id,
            subreddit="subreddit",
            author="author",
        )

    def test_dfs(self):
        root = 1
        children = {1:[2,3], 2:[5,6], 6:[7]}
        dfs = create_data_v2.DFS(children)
        paths = dfs.get_all_paths(root)
        assert paths == [[1,2,5], [1,2,6,7], [1,3]]


    def create_examples(self):
        with open("reddit/testdata/simple_thread.json") as f:
            thread = json.loads(f.read())

        thread = [
            create_data_v2.normalise_comment(comment) for comment in thread
        ]

        examples = list(create_data_v2.create_examples(thread, False, False, None))
        expected_examples = [
            {
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>CCCC",
            },
            {
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>DDDD<sep>EEEE",
            },
            {
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>DDDD<sep>too long to create an example",
            },      
            {
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>AAAA<sep>BBBB<sep>DDDD<sep>123",
            },
            {
                'subreddit': "subreddit-A",
                'thread_id': "testthread",
                'comments': "[submission]<sep>FFFF",
            }]
        self.assertCountEqual(examples, expected_examples)


if __name__ == "__main__":
    unittest.main()
