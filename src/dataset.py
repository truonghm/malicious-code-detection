from __future__ import annotations

import os
import random
import warnings
from typing import Any, Callable, Generator, Iterator, List, Optional, Tuple

import chardet
import esprima
from calmjs.parse.exceptions import ECMASyntaxError
from calmjs.parse.lexers.es5 import Lexer as ES5Lexer
from gensim import utils
from gensim.corpora.textcorpus import TextDirectoryCorpus, walk
from slimit.lexer import Lexer


def tokenize_calmjs(snippet: str) -> List[Tuple[Any, Any]]:
    try:
        lexer = Lexer()
        lexer.input(snippet)
        tokens = []
        for token in lexer:
            tokens.append((token.type, token.value))
        return tokens
    except ECMASyntaxError as e:
        raise


def tokenize_esprima(snippet: str) -> List[Tuple[Any, Any]]:
    tokens = []
    try:
        result = esprima.tokenize(snippet)
        for token in result:
            tokens.append((token.type, token.value))
    except Exception as e:
        raise
    return tokens


class CodeDirectoryCorpus(TextDirectoryCorpus):  # type: ignore
    def __init__(self, input: str, **kwargs: Any) -> None:
        self.sampling_percentage = kwargs.pop("sampling", 100)
        self.seed = kwargs.pop("seed", None)
        self.labels: List[int] = []
        self.length_: int = 0
        super().__init__(input, **kwargs)
        self.filepaths = [p for p in self.iter_filepaths()]

    def iter_filepaths(self) -> Generator[str, None, None]:
        if self.seed is not None:
            random.seed(self.seed)
        for depth, dirpath, dirnames, filenames in walk(self.input):
            if self.min_depth <= depth <= self.max_depth:
                if self.pattern is not None:
                    filenames = (n for n in filenames if self.pattern.match(n) is not None)
                if self.exclude_pattern is not None:
                    filenames = (n for n in filenames if self.exclude_pattern.search(os.path.join(dirpath, n)) is None)
                filenames = list(filenames)
                sampled_filenames = random.sample(filenames, int(len(filenames) * self.sampling_percentage / 100))
                for name in sampled_filenames:
                    self.length_ += 1
                    yield os.path.join(dirpath, name)

    def getstream(self) -> Generator[Tuple[str, str, int], None, None]:
        num_texts = 0
        for path in self.filepaths:
            try:
                with utils.open(path, "rb") as f:
                    rawdata = f.read()
                    result = chardet.detect(rawdata)
                    encoding = result["encoding"]
                    if not encoding:
                        encoding = "utf-8"
                    text = rawdata.decode(encoding)
                    if "bad" in os.path.dirname(path):
                        label = 1
                    else:
                        label = 0
                    self.labels.append(label)
                    yield text, path, label
                    num_texts += 1
                # self.length_ += 1
            except UnicodeDecodeError:
                print(f"Error decoding file: {path}")
                continue

    def get_texts(
        self, tokenizer: Callable = tokenize_esprima, label: Optional[int] = None
    ) -> Generator[CodeSnippet, None, None]:
        for snippet, path, label_ in self.getstream():
            # yield [word for word in snippet.split()]
            if not label:
                code_snippet = CodeSnippet(snippet, path, label_, tokenizer)
                if code_snippet.tokens:
                    yield code_snippet
            elif label == 1:
                if "bad" in os.path.dirname(path):
                    code_snippet = CodeSnippet(snippet, path, label, tokenizer)
                    if code_snippet.tokens:
                        yield code_snippet
            elif label == 0:
                if "good" in os.path.dirname(path):
                    code_snippet = CodeSnippet(snippet, path, label, tokenizer)
                    if code_snippet.tokens:
                        yield code_snippet
            else:
                raise ValueError(f"Invalid label: {label}")

    def __len__(self) -> int:
        # self.length = sum(1 for _ in self.get_texts())
        warnings.warn(
            "This is only an approximate length, since there might be errors in the corpus leading to some files not being read",  # noqa: E501
            RuntimeWarning,
        )
        return self.length_

    def init_dictionary(self, dictionary: Any) -> None:
        warnings.warn("Disabled dictionary initialization", RuntimeWarning)


class CodeSnippet:
    def __init__(self, snippet: str, path: str, label: int, tokenizer: Callable) -> None:
        self.snippet = snippet
        try:
            self.tokens = tokenizer(snippet)
        except Exception as e:
            print(f"{path}: {e}")
            self.tokens = None
        self.path = path
        self.label = label

    def __str__(self) -> str:
        return str(self.snippet)

    def __iter__(self) -> Iterator[str]:
        return iter(self.snippet)
