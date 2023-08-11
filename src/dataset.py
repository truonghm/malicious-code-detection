from __future__ import annotations

import os
import random
from typing import Any, Generator, Iterator, List, Tuple

import chardet
from gensim import utils
from gensim.corpora.textcorpus import TextDirectoryCorpus, walk


class CodeDirectoryCorpus(TextDirectoryCorpus):  # type: ignore
    def __init__(self, input: str, **kwargs: Any) -> None:
        self.sampling_percentage = kwargs.pop("sampling", 100)
        self.seed = kwargs.pop("seed", None)
        self.labels: List[int] = []
        self.length_: int = 0
        super().__init__(input, **kwargs)

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
                    yield os.path.join(dirpath, name)

    def getstream(self) -> Generator[Tuple[str, str, int], None, None]:
        num_texts = 0
        for path in self.iter_filepaths():
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
                self.length_ += 1
            except UnicodeDecodeError:
                print(f"Error decoding file: {path}")
                continue

    def get_texts(self) -> Generator[CodeSnippet, None, None]:
        for snippet, path, label in self.getstream():
            # yield [word for word in snippet.split()]
            yield CodeSnippet(snippet, path, label)

    def __len__(self) -> int:
        # self.length = sum(1 for _ in self.get_texts())
        return self.length_


class CodeSnippet:
    def __init__(self, snippet: str, path: str, label: int) -> None:
        self.raw_snippet = snippet
        self.snippet = [word for word in snippet.split()]
        self.path = path
        self.label = label

    def __str__(self) -> str:
        return str(self.snippet)

    def __iter__(self) -> Iterator[str]:
        return iter(self.snippet)
