"""Tokenizer and vocabulary for ToM benchmark."""

CATS = [f"k{i}" for i in range(80)]
ENTS = [f"e{i}" for i in range(60)]
EVTS = [f"v{i}" for i in range(50)]
PERS = [f"n{i}" for i in range(20)]
OBJS = [f"o{i}" for i in range(15)]
LOCS = [f"l{i}" for i in range(12)]


class SimpleTokenizer:
    SPECIAL = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<sep>": 3,
               "<cot>": 4, "<ans>": 5, "<unk>": 6}

    def __init__(self):
        self.t2i = dict(self.SPECIAL)
        self.i2t = {v: k for k, v in self.t2i.items()}
        self._n = len(self.SPECIAL)

    def add(self, tokens):
        for t in (tokens if isinstance(tokens, list) else [tokens]):
            if t not in self.t2i:
                self.t2i[t] = self._n
                self.i2t[self._n] = t
                self._n += 1

    def encode(self, text):
        return [1] + [self.t2i.get(w, 6) for w in text.strip().split()] + [2]

    def decode(self, ids):
        return " ".join(self.i2t.get(i, "?") for i in ids
                        if i not in self.SPECIAL.values())

    @property
    def vocab_size(self):
        return self._n

    @property
    def pad_id(self):
        return 0


def make_tokenizer():
    tok = SimpleTokenizer()
    tok.add([
        "all", "are", "is", "a", "not", "yes", "no", ".", "?",
        "causes", "does", "cause", "puts", "the", "in", "leaves",
        "room", "moves", "to", "returns", "where", "think", "sees",
        "after", "and", "but", "some", "with", "also", "because",
        "since", "therefore", "so", "often", "together", "seen",
        "observed", "before", "saw", "left", "last", "knows",
        "enters", "talks", "happens", "when", "occurs", "near",
        "tells", "frequently",
    ])
    tok.add(CATS + ENTS + EVTS + PERS + OBJS + LOCS)
    return tok
