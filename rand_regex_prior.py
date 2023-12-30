"""
Download, preprocess and serve the TinyStories dataset as a DataLoader.
"""

import argparse
import glob
import json
import os
import random
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import torch
import numpy as np
import requests
import sentencepiece as spm
import torch
import torch.distributed as dist
from tqdm import tqdm

from tokenizer import Tokenizer

import random
from collections import UserString, UserList
from abc import ABCMeta, abstractmethod

import string
import re


def numpy_geometric_sample(limit, p=0.3):
    # Sample using NumPy's geometric function
    if (1 - p) ** limit > .1:
        raise ValueError()
    sample = limit + 1
    while sample > limit:
        sample = np.random.geometric(p) - 1  # Subtract 1 to start counting from 0

    # Ensure the sample does not exceed the limit
    return sample


# Get all ASCII characters
ascii_chars = list(string.printable + string.ascii_letters)
ascii_chars.remove('\n')

# Escape special characters
escaped_chars_mapping = {char: re.escape(char) for char in ascii_chars}
escaped_ascii_chars = [re.escape(char) for char in ascii_chars]

string_len_sampler = lambda: numpy_geometric_sample(10, p=.3) + 1
star_len_sampler = lambda: numpy_geometric_sample(10, p=.3)
num_examples_sampler = lambda: random.randint(1, 10)

max_depth = 4

vocab = ['<SOS>', '<EOS>', '<SEP>', '<SOP>', '<PAD>']  # used like <SOS>REGEX_SAMPLE<SEP>REGEX_SAMPLE<SEP>...<SOP>REGEX<EOS>
vocab += escaped_ascii_chars + ascii_chars

vocab_inverse = {c: i for i, c in enumerate(vocab)}


class Term(metaclass=ABCMeta):
    @abstractmethod
    def sample_tokenized(self):
        pass  # should return random instances that are matched by this term as vocab indice list

    @abstractmethod
    def get_tokenized_regex(self):
        pass

    def __repr__(self):
        return "'" + str(self) + "'"

    def __str__(self):
        return self.tokenized_to_string(self.get_tokenized_regex())

    @staticmethod
    def tokenized_to_string(tokenized_list):
        return ''.join(vocab[i] for i in tokenized_list)

    def sample(self):
        return self.tokenized_to_string(self.sample_tokenized())


class String(Term):
    def __init__(self, depth=0):
        self.wildcard = '.'
        self.indices = random.choices([0] * len(ascii_chars) + list(range(1, len(ascii_chars) + 1)),
                                      k=string_len_sampler())

    def get_tokenized_regex(self):
        full_escaped_alphabet = [self.wildcard] + escaped_ascii_chars
        return [vocab_inverse[full_escaped_alphabet[i]] for i in self.indices]

    def sample_tokenized(self):
        return [vocab_inverse[random.choice(ascii_chars)] if i == 0 else vocab_inverse[ascii_chars[i - 1]] for i in
                self.indices]

    def sample(self):
        return ''.join(vocab[i] for i in self.sample_tokenized())


class Or(Term):
    def __init__(self, depth=0):
        self.children = [get_random_term(depth=depth + 1), get_random_term(depth=depth + 1)]

    def get_tokenized_regex(self):
        return [vocab_inverse['(']] + self.children[0].get_tokenized_regex() + [vocab_inverse['|']] + self.children[
            1].get_tokenized_regex() + [vocab_inverse[')']]

    def sample_tokenized(self):
        return random.choice(self.children).sample_tokenized()


class UnnecessaryParentheses(Term):
    def __init__(self, depth=0):
        self.content = get_random_term(exclude_classes=[UnnecessaryParentheses], depth=depth + 1)

    def get_tokenized_regex(self):
        return [vocab_inverse['(']] + self.content.get_tokenized_regex() + [vocab_inverse[')']]

    def sample_tokenized(self):
        return self.content.sample_tokenized()


class Star(Term):
    def __init__(self, depth=0):
        self.content = get_random_term(exclude_classes=[Star], depth=depth + 1)

    def get_tokenized_regex(self):
        return [vocab_inverse['(']] + self.content.get_tokenized_regex() + [vocab_inverse[')'], vocab_inverse['*']]

    def sample_tokenized(self):
        return sum([self.content.sample_tokenized() for _ in range(star_len_sampler())], [])


class Concat(Term):
    def __init__(self, depth=0):
        self.children = get_random_term(depth=depth + 1), get_random_term(depth=depth + 1)

    def get_tokenized_regex(self):
        return sum([c.get_tokenized_regex() for c in self.children], [])

    def sample_tokenized(self):
        return sum([c.sample_tokenized() for c in self.children], [])


def get_random_term(exclude_classes=[], depth=0):
    classes = [cl for cl in [String, String, Or, Star, Concat, UnnecessaryParentheses] if cl not in exclude_classes]  # String is more likely
    if depth > max_depth:
        classes = [String]
    return random.choice(classes)(depth=depth + 1)


def sample_task(max_len=1000, num_examples_sampler=num_examples_sampler):
    while True:
        t = get_random_term()
        num_examples = num_examples_sampler()
        examples = sum([t.sample_tokenized() + [vocab_inverse['<SEP>']] for _ in range(num_examples)], [])
        examples[-1] = vocab_inverse['<SOP>']
        full_string = [vocab_inverse['<SOS>']] + examples + t.get_tokenized_regex() + [vocab_inverse['<EOS>']]
        if len(full_string) <= max_len:
            return full_string


class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, max_len=1000, num_examples_sampler=num_examples_sampler):
        self.max_len = max_len
        self.num_examples_sampler = num_examples_sampler

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        while True:
            sampled_task = sample_task(self.max_len, self.num_examples_sampler)
            yield torch.tensor(sampled_task[:-1] + [vocab_inverse['<PAD>']] * (self.max_len+1-len(sampled_task))),\
                torch.tensor(sampled_task[1:] + [vocab_inverse['<PAD>']] * (self.max_len+1-len(sampled_task)))


class Task:
    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, vocab_size=len(vocab), max_seq_len=1000,  **rest_kwargs):
        assert vocab_size == len(vocab)
        print('rest', rest_kwargs)
        ds = Dataset(max_len=max_seq_len)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y

PAD_IDX = vocab_inverse['<PAD>']
# -----------------------------------------------------------------------------
# CLI for constructing the dataset

if __name__ == "__main__":
    """
    These stages are designed to be run in order.

    To tokenize data with the Llama 2 tokenizer:
    python tinystories.py download
    python tinystories.py pretokenize

    To tokenize data with a custom tokenizer we train ourselves with sentencepiece, e.g.:
    python tinystories.py download
    python tinystories.py train_vocab --vocab_size=2048
    python tinystories.py pretokenize --vocab_size=2048
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["download", "pretokenize", "train_vocab"])
    parser.add_argument("--vocab_size", type=int, default=0, help="pretokenization vocab size. 0 = use Llama 2 tokenizer.")
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    if args.stage == "download":
        download()
    elif args.stage == "train_vocab":
        train_vocab(vocab_size=args.vocab_size)
    elif args.stage == "pretokenize":
        pretokenize(vocab_size=args.vocab_size)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
