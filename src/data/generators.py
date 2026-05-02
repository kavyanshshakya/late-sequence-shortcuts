"""Theory of Mind story generators."""

import random
from .tokenizer import PERS, OBJS, LOCS, make_tokenizer
from .dataset import Example


class HardToMGenerator:
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.tokenizer = make_tokenizer()
        self._train = PERS[:16]
        self._holdout = PERS[16:]

    def _dist(self, persons, obj):
        temps = [
            f"{self.rng.choice(persons)} enters the room",
            f"{self.rng.choice(persons)} talks to {self.rng.choice(persons)}",
            f"{self.rng.choice(persons)} leaves the room",
            f"{self.rng.choice(persons)} sees {self.rng.choice(OBJS)}",
            f"{self.rng.choice(persons)} puts the {self.rng.choice(OBJS)} in {self.rng.choice(LOCS)}",
        ]
        return self.rng.choice(temps)

    def _gen(self, n, sc, dist_range, pool):
        is_clean = (sc <= 0.5)
        examples = []
        for _ in range(n):
            pl = self.rng.sample(list(pool), min(3, len(pool)))
            p1 = pl[0]
            p2 = pl[1] if len(pl) > 1 else pl[0]
            obj = self.rng.choice(OBJS)
            ls = self.rng.sample(LOCS, min(4, len(LOCS)))
            la, lb, lc = ls[0], ls[1], ls[2]
            ld = ls[3] if len(ls) > 3 else ls[0]
            nd = self.rng.randint(dist_range[0], dist_range[1])
            sc_type = self.rng.randint(0, 4)

            if sc_type == 0:
                parts = [
                    f"{p1} puts the {obj} in {la}",
                    f"the {obj} moves to {lb}",
                    f"{p1} sees the {obj} in {lb}",
                    f"{p1} leaves the room",
                    f"the {obj} moves to {lc}",
                    f"{p1} returns to the room",
                ]
                question = f"where does {p1} think the {obj} is ?"
                answer = lb; label = 0
                cot = f"{p1} saw {lb} missed {lc}"
                rt = "2move_fb"; nh = 3
            elif sc_type == 1:
                parts = [
                    f"{p1} puts the {obj} in {la}",
                    f"the {obj} moves to {lb}",
                    f"{p1} sees the {obj} in {lb}",
                    f"{p1} leaves the room",
                    f"the {obj} moves to {lc}",
                    f"the {obj} moves to {ld}",
                    f"{p1} returns to the room",
                ]
                question = f"where does {p1} think the {obj} is ?"
                answer = lb; label = 0
                cot = f"{p1} last saw {lb}"
                rt = "3move_fb"; nh = 4
            elif sc_type == 2:
                parts = [
                    f"{p2} puts the {obj} in {la}",
                    f"{p1} sees {p2} put the {obj} in {la}",
                    f"{p2} leaves the room",
                    f"the {obj} moves to {lb}",
                    f"{p1} sees the {obj} in {lb}",
                    f"{p2} returns to the room",
                ]
                question = f"where does {p1} think {p2} think the {obj} is ?"
                answer = la; label = 0
                cot = f"{p1} knows {p2} missed move"
                rt = "2nd_order"; nh = 4
            elif sc_type == 3:
                parts = [
                    f"{p1} puts the {obj} in {la}",
                    f"the {obj} moves to {lb}",
                    f"{p1} sees the {obj} in {lb}",
                    f"the {obj} moves to {lc}",
                    f"{p1} sees the {obj} in {lc}",
                ]
                question = f"where does {p1} think the {obj} is ?"
                answer = lc; label = 1
                cot = f"{p1} saw all moves"
                rt = "true_2move"; nh = 2
            else:
                parts = [
                    f"{p1} puts the {obj} in {la}",
                    f"{p1} leaves the room",
                    f"{p2} moves the {obj} to {lb}",
                    f"{p1} returns to the room",
                ]
                question = f"where does {p1} think the {obj} is ?"
                answer = la; label = 0
                cot = f"{p1} left before move"
                rt = "classic_fb"; nh = 2

            for _ in range(nd):
                pos = self.rng.randint(1, max(1, len(parts) - 1))
                parts.insert(pos, self._dist(pl, obj))

            if not is_clean:
                if self.rng.random() < sc:
                    parts.append(f"the {answer} is near the room")
                    cue = "hint"
                else:
                    cue = "no_hint"
            else:
                cue = None

            prem = " . ".join(parts) + " ."
            examples.append(Example(
                premises=prem, question=question, answer=answer,
                label=label, cot=cot, shortcut_available=not is_clean,
                reasoning_type=rt, n_hops=nh, n_distractors=nd,
                shortcut_cue=cue,
            ))

        self.rng.shuffle(examples)
        return examples

    def generate_clean(self, n, dist_range=(4, 8)):
        return self._gen(n, 0.5, dist_range, self._train)

    def generate_shortcut(self, n, correlation=0.9, dist_range=(4, 8)):
        return self._gen(n, correlation, dist_range, self._train)

    def generate_memorization_test(self, n):
        return self._gen(n, 0.5, (4, 8), self._holdout)


class HardToMGeneratorPositional(HardToMGenerator):
    """Can place shortcut at beginning or end of narrative."""

    def generate_shortcut_at_position(self, n, correlation=0.9,
                                       position="end", dist_range=(4, 8)):
        examples = []
        for _ in range(n):
            pl = self.rng.sample(list(self._train), min(3, len(self._train)))
            p1 = pl[0]
            p2 = pl[1] if len(pl) > 1 else pl[0]
            obj = self.rng.choice(OBJS)
            ls = self.rng.sample(LOCS, min(4, len(LOCS)))
            la, lb, lc = ls[0], ls[1], ls[2]
            ld = ls[3] if len(ls) > 3 else ls[0]
            nd = self.rng.randint(dist_range[0], dist_range[1])
            sc_type = self.rng.randint(0, 4)

            if sc_type == 0:
                parts = [
                    f"{p1} puts the {obj} in {la}",
                    f"the {obj} moves to {lb}",
                    f"{p1} sees the {obj} in {lb}",
                    f"{p1} leaves the room",
                    f"the {obj} moves to {lc}",
                    f"{p1} returns to the room",
                ]
                question = f"where does {p1} think the {obj} is ?"
                answer = lb; label = 0; rt = "2move_fb"; nh = 3
            elif sc_type == 1:
                parts = [
                    f"{p1} puts the {obj} in {la}",
                    f"the {obj} moves to {lb}",
                    f"{p1} sees the {obj} in {lb}",
                    f"{p1} leaves the room",
                    f"the {obj} moves to {lc}",
                    f"the {obj} moves to {ld}",
                    f"{p1} returns to the room",
                ]
                question = f"where does {p1} think the {obj} is ?"
                answer = lb; label = 0; rt = "3move_fb"; nh = 4
            elif sc_type == 2:
                parts = [
                    f"{p2} puts the {obj} in {la}",
                    f"{p1} sees {p2} put the {obj} in {la}",
                    f"{p2} leaves the room",
                    f"the {obj} moves to {lb}",
                    f"{p1} sees the {obj} in {lb}",
                    f"{p2} returns to the room",
                ]
                question = f"where does {p1} think {p2} think the {obj} is ?"
                answer = la; label = 0; rt = "2nd_order"; nh = 4
            elif sc_type == 3:
                parts = [
                    f"{p1} puts the {obj} in {la}",
                    f"the {obj} moves to {lb}",
                    f"{p1} sees the {obj} in {lb}",
                    f"the {obj} moves to {lc}",
                    f"{p1} sees the {obj} in {lc}",
                ]
                question = f"where does {p1} think the {obj} is ?"
                answer = lc; label = 1; rt = "true_2move"; nh = 2
            else:
                parts = [
                    f"{p1} puts the {obj} in {la}",
                    f"{p1} leaves the room",
                    f"{p2} moves the {obj} to {lb}",
                    f"{p1} returns to the room",
                ]
                question = f"where does {p1} think the {obj} is ?"
                answer = la; label = 0; rt = "classic_fb"; nh = 2

            for _ in range(nd):
                pos = self.rng.randint(1, max(1, len(parts) - 1))
                parts.insert(pos, self._dist(pl, obj))

            if self.rng.random() < correlation:
                s = f"the {answer} is near the room"
                if position == "begin":
                    parts.insert(0, s)
                else:
                    parts.append(s)
                cue = f"hint_{position}"
            else:
                cue = f"no_hint_{position}"

            prem = " . ".join(parts) + " ."
            examples.append(Example(
                premises=prem, question=question, answer=answer,
                label=label, cot=None, shortcut_available=True,
                reasoning_type=rt, n_hops=nh, n_distractors=nd,
                shortcut_cue=cue,
            ))

        self.rng.shuffle(examples)
        return examples

    def generate_shortcut_at_fraction(self, n, correlation=0.9,
                                       position_frac=1.0, dist_range=(4, 8)):
        """Insert shortcut at fractional position (0.0=begin, 1.0=end).

        Used for Experiment H 9-position sweep: tests positions 0%, 25%, 50%,
        75%, 80%, 85%, 90%, 95%, 100% of narrative length.
        """
        assert 0.0 <= position_frac <= 1.0
        # Call generate_shortcut_at_position to build base narrative structure,
        # then adjust cue insertion to fractional position.
        # For simplicity, we re-implement here with same templates.
        examples = []
        for _ in range(n):
            pl = self.rng.sample(list(self._train), min(3, len(self._train)))
            p1 = pl[0]
            p2 = pl[1] if len(pl) > 1 else pl[0]
            obj = self.rng.choice(OBJS)
            ls = self.rng.sample(LOCS, min(4, len(LOCS)))
            la, lb, lc = ls[0], ls[1], ls[2]
            ld = ls[3] if len(ls) > 3 else ls[0]
            nd = self.rng.randint(dist_range[0], dist_range[1])
            sc_type = self.rng.randint(0, 4)

            if sc_type == 0:
                parts = [
                    f"{p1} puts the {obj} in {la}",
                    f"the {obj} moves to {lb}",
                    f"{p1} sees the {obj} in {lb}",
                    f"{p1} leaves the room",
                    f"the {obj} moves to {lc}",
                    f"{p1} returns to the room",
                ]
                question = f"where does {p1} think the {obj} is ?"
                answer = lb; label = 0; rt = "2move_fb"; nh = 3
            elif sc_type == 1:
                parts = [
                    f"{p1} puts the {obj} in {la}",
                    f"the {obj} moves to {lb}",
                    f"{p1} sees the {obj} in {lb}",
                    f"{p1} leaves the room",
                    f"the {obj} moves to {lc}",
                    f"the {obj} moves to {ld}",
                    f"{p1} returns to the room",
                ]
                question = f"where does {p1} think the {obj} is ?"
                answer = lb; label = 0; rt = "3move_fb"; nh = 4
            elif sc_type == 2:
                parts = [
                    f"{p2} puts the {obj} in {la}",
                    f"{p1} sees {p2} put the {obj} in {la}",
                    f"{p2} leaves the room",
                    f"the {obj} moves to {lb}",
                    f"{p1} sees the {obj} in {lb}",
                    f"{p2} returns to the room",
                ]
                question = f"where does {p1} think {p2} think the {obj} is ?"
                answer = la; label = 0; rt = "2nd_order"; nh = 4
            elif sc_type == 3:
                parts = [
                    f"{p1} puts the {obj} in {la}",
                    f"the {obj} moves to {lb}",
                    f"{p1} sees the {obj} in {lb}",
                    f"the {obj} moves to {lc}",
                    f"{p1} sees the {obj} in {lc}",
                ]
                question = f"where does {p1} think the {obj} is ?"
                answer = lc; label = 1; rt = "true_2move"; nh = 2
            else:
                parts = [
                    f"{p1} puts the {obj} in {la}",
                    f"{p1} leaves the room",
                    f"{p2} moves the {obj} to {lb}",
                    f"{p1} returns to the room",
                ]
                question = f"where does {p1} think the {obj} is ?"
                answer = la; label = 0; rt = "classic_fb"; nh = 2

            for _ in range(nd):
                pos = self.rng.randint(1, max(1, len(parts) - 1))
                parts.insert(pos, self._dist(pl, obj))

            if self.rng.random() < correlation:
                s = f"the {answer} is near the room"
                insert_idx = int(round(position_frac * len(parts)))
                insert_idx = max(0, min(insert_idx, len(parts)))
                parts.insert(insert_idx, s)
                cue = f"hint_frac{int(position_frac*100)}"
            else:
                cue = f"no_hint_frac{int(position_frac*100)}"

            prem = " . ".join(parts) + " ."
            examples.append(Example(
                premises=prem, question=question, answer=answer,
                label=label, cot=None, shortcut_available=True,
                reasoning_type=rt, n_hops=nh, n_distractors=nd,
                shortcut_cue=cue,
            ))

        self.rng.shuffle(examples)
        return examples
