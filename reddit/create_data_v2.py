"""A Dataflow script for creating datasets from reddit.

For usage see README.md.
"""

import argparse
import json
import logging
import os
import re
from collections import defaultdict
from functools import partial

import apache_beam as beam
import tensorflow as tf
from apache_beam import pvalue
from apache_beam.io import BigQuerySource, Read
from apache_beam.io.textio import WriteToText
from apache_beam.io.tfrecordio import WriteToTFRecord
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
import cld3

_TF_FORMAT = "TF"
_JSON_FORMAT = "JSON"


def _parse_args(argv=None):
    """Parse command line arguments."""

    def _nonnegative_int(value):
        """Define a non-negative integer ArgumentParser type."""
        value = int(value)
        if value < 0:
            raise argparse.ArgumentTypeError(
                "Value must be non-negative, {} was passed.".format(value)
            )
        return value

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reddit_table",
        required=True,
        help="The BigQuery table to read comments from, in " "project:table format.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Google cloud storage output directory to write the dataset.",
    )
    parser.add_argument(
        "--dataset_format",
        choices={_TF_FORMAT, _JSON_FORMAT},
        default="TF",
        help="The dataset format to write. 'TF' for serialized tensorflow "
        "examples in TFRecords. 'JSON' for text files with one JSON "
        "object per line.",
    )
    parser.add_argument(
        "--num_shards",
        default=0,
        type=_nonnegative_int,
        help="The number of shards each dataset.",
    )
    parser.add_argument(
        "--skip_single_comment",
        default=False,
        type=bool,
        help="Skip dialogue tree if there is only single comment.",
    )
    parser.add_argument(
        "--detect_lang",
        default=False,
        type=bool,
        help="Do language detection with fasttext lid.176.bin model.",
    )

    return parser.parse_known_args(argv)


def normalise_comment(comment):
    """Create a _Comment object from a row in the BigQuery table."""

    # Represent a reddit comment.
    def _normalise_id(raw_id):
        """Reddit IDs start with t1_, t2_, etc. which need to be stripped."""
        return re.sub("^t[0-9]_", "", raw_id)

    return dict(
        id=comment["id"],
        thread_id=_normalise_id(comment["link_id"]),
        parent_id=_normalise_id(comment["parent_id"]),
        body=comment["body"],
        author=comment["author"],
        subreddit=comment["subreddit"],
    )


class DFS:
    "Get all paths in a generic tree starting from the root"

    def __init__(self, children):
        self.paths = []
        self.children = children

    def _dfs(self, node, path):
        path.append(node)
        if node not in self.children:
            self.paths.append(path)
        else:
            for child in self.children[node]:
                self._dfs(child, path.copy())

    def get_all_paths(self, root):
        self._dfs(root, [])
        return self.paths


def generate_paths_for_thread(thread_comments):
    children = defaultdict(list)
    for comment in thread_comments:
        children[comment["parent_id"]].append(comment["id"])
    reddit_dfs = DFS(children)
    root = thread_comments[0]["thread_id"]
    reddit_paths = reddit_dfs.get_all_paths(root)
    return reddit_paths


def create_examples(thread, skip_single_comment, detect_lang, ft_langmodel=None):
    """Creates serialized tensorflow examples from a reddit thread."""
    thread_comments = list(thread)
    id_to_comment = {comment["id"]: comment for comment in thread_comments}

    # generate all dialogue paths
    paths = generate_paths_for_thread(thread_comments)

    # iterate each path to generate text dialogues
    for path in paths:
        # if it's a single comment skip
        if skip_single_comment and len(path) <= 2:
            continue

        path_comment_texts = []
        for i, comment_id in enumerate(path):
            if i == 0:
                comment_text = "[submission]"
            else:
                comment_text = id_to_comment[comment_id]["body"]

            if comment_text not in ["[deleted]", "[removed]"]:
                path_comment_texts.append(comment_text)

        # check again after removals; if it's a single comment skip
        if skip_single_comment and len(path_comment_texts) <= 2:
            continue

        example = {}
        example["subreddit"] = thread_comments[0]["subreddit"]
        example["thread_id"] = thread_comments[0]["thread_id"]
        example["comments"] = "<sep>".join(path_comment_texts)
        if detect_lang:
            text = " ".join(path_comment_texts)
            # fasttext
            # example['language'] = ft_langmodel.predict(text.replace(
            #     "\n", " "))[0][0][9:]
            # cld3
            example["language"] = cld3.get_language(text).language
            # # langdetect
            # example['language'] = langdetect.detect(text)

        yield example


def _features_to_serialized_tf_example(features):
    """Convert a string dict to a serialized TF example.

    The dictionary maps feature names (strings) to feature values (strings).
    """
    example = tf.train.Example()
    for feature_name, feature_value in features.items():
        example.features.feature[feature_name].bytes_list.value.append(
            feature_value.encode("utf-8")
        )
    return example.SerializeToString()


# def _shuffle(pcollection):
#     """Shuffles the input pcollection."""
#     pcollection |= "add random key" >> beam.Map(lambda value:
#                                                 (uuid.uuid4(), value))
#     pcollection |= "group by key" >> beam.GroupByKey()
#     pcollection |= "get shuffled values" >> beam.FlatMap(lambda t: t[1])
#     return pcollection


class _LanguageSplitFn(beam.DoFn):
    """Splits an input PCollection of examples into different languages.
    If language detection is not enabled ALL_TAG will be used.
    """

    ALL_TAG = "all"
    EN_TAG = "en"
    TR_TAG = "tr"
    OTHER_TAG = "other"

    def __init__(self, detect_lang):
        self.detect_lang = detect_lang

    def process(self, example):
        if self.detect_lang:
            if example["language"] == "en":
                yield pvalue.TaggedOutput(self.EN_TAG, example)
            elif example["language"] == "tr":
                yield pvalue.TaggedOutput(self.TR_TAG, example)
            else:
                yield pvalue.TaggedOutput(self.OTHER_TAG, example)
        else:
            yield pvalue.TaggedOutput(self.ALL_TAG, example)


def run(argv=None, comments=None):
    """Run the beam pipeline.

    Args:
        argv: (optional) the command line flags to parse.
        comments_collection: (optional) a list of comment JSON objects to
            process. Used in unit-tests to avoid requiring a BigQuery source.
    """
    args, pipeline_args = _parse_args(argv)

    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True

    with beam.Pipeline(options=pipeline_options) as p:
        # if args.detect_lang:
        #     import fasttext
        #     ft_langmodel = fasttext.load_model(
        #         "/Users/keremturgutlu/Desktop/git/fastText/models/lid.176.bin")
        # else:
        #     ft_langmodel = None
        ft_langmodel = None

        # print(args, pipeline_args)
        # raise Exception("stop")

        if comments is not None:
            comments = p | ("Read in-memory comments") >> beam.Create(comments)

        elif "DirectRunner" in pipeline_args or "PortableRunner" in pipeline_args:
            with open("reddit/testdata/simple_thread.json") as f:
                comments = json.loads(f.read())
            comments = p | ("Read in-memory comments") >> beam.Create(comments)

        else:
            comments = p | ("Read " + args.reddit_table) >> Read(
                BigQuerySource(args.reddit_table)
            )

        # Normalize and Create a _Comment:namedtuple  object from a row in the BigQuery table.
        comments |= "Normalise comments" >> beam.Map(normalise_comment)

        # Create (thread_id,_Comment) : (k,v) pairs
        thread_id_to_comments = comments | (
            "Key by thread id"
            >> beam.Map(lambda comment: (comment["thread_id"], comment))
        )

        # Group Comments by thread id
        threads = thread_id_to_comments | (
            "Group comments by thread ID" >> beam.GroupByKey()
        )

        # Get threads
        threads = threads | ("Get threads" >> beam.Map(lambda t: t[1]))

        # Generate dialogue trees (examples) from threads
        examples = threads | (
            "Create {} examples".format(args.dataset_format)
            >> beam.FlatMap(
                partial(
                    create_examples,
                    skip_single_comment=args.skip_single_comment,
                    detect_lang=args.detect_lang,
                    ft_langmodel=ft_langmodel,
                )
            )
        )

        # examples = _shuffle(examples)

        examples |= "split by language" >> beam.ParDo(
            _LanguageSplitFn(args.detect_lang)
        ).with_outputs(
            _LanguageSplitFn.ALL_TAG,
            _LanguageSplitFn.EN_TAG,
            _LanguageSplitFn.TR_TAG,
            _LanguageSplitFn.OTHER_TAG,
        )

        if args.dataset_format == _JSON_FORMAT:
            write_sink = WriteToText
            file_name_suffix = ".json"
            serialize_fn = json.dumps
        else:
            assert args.dataset_format == _TF_FORMAT
            write_sink = WriteToTFRecord
            file_name_suffix = ".tfrecord"
            serialize_fn = _features_to_serialized_tf_example

        if args.detect_lang:
            dataset_tags = [
                ("en", _LanguageSplitFn.EN_TAG),
                ("tr", _LanguageSplitFn.TR_TAG),
                ("other", _LanguageSplitFn.OTHER_TAG),
            ]
        else:
            dataset_tags = [("all", _LanguageSplitFn.ALL_TAG)]

        for name, tag in dataset_tags:
            serialized_examples = examples[tag] | (
                "serialize {} examples".format(name) >> beam.Map(serialize_fn)
            )
            (
                serialized_examples
                | ("write " + name)
                >> write_sink(
                    os.path.join(args.output_dir, name),
                    file_name_suffix=file_name_suffix,
                    num_shards=args.num_shards,
                )
            )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
