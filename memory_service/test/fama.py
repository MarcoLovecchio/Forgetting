"""FAMA: how much the memory remembers, and how much it really forgets.

The metric comes from Memora (Uddin et al., *From Recall to Forgetting:
Benchmarking Long-Term Memory for Personalized Agents*, ACL 2026,
arXiv:2604.20006). For each evaluation question two families of binary
criteria are written:

    memory_presence     the valid information the answer MUST contain
    forgetting_absence  the superseded information it must NOT contain

and from them a single score is derived:

    MPA  = fraction of presence criteria satisfied
    FAA  = fraction of absence criteria satisfied
    lam  = N_forget / (N_presence + N_forget)
    FAMA = max(0, MPA - lam * (1 - FAA))

The weight `lam` makes a question that is all about forgetting be judged all on
forgetting: with N_presence = 0 the formula degenerates into FAMA = FAA, and
with N_forget = 0 into FAMA = MPA. The `max` keeps the score in [0, 1].

Why it is needed here
---------------------
FAMA is born as a metric **on the answer**, and the answer is the last link of a
long chain: consolidation writes the store, retrieval decides whether to query
the archive, generate_answer composes. A single number on that chain never says
whose fault it is - and this test has already seen it happen, with the delete of
the address consolidated correctly and the answer still saying where the user
lived.

That is why here the criteria are applied **twice**, the same criteria with the
same formula, but to two different objects:

    FAMA-answer     on the text composed by generate_answer
    FAMA-memory     on the active memories, right after consolidating the
                    message that precedes the question (memory_snapshot_turn)

The second is the one that speaks about consolidation, and it is deterministic:
the store is structured, `status` already tells the active from the deleted, no
judge is needed. The gap between the two isolates the culprit:

    memory high + answer high      all good
    memory high + answer low       fault of generate_answer or of retrieval
    memory low  + answer low       fault of consolidation
    memory low  + answer high      if the memory missed a fact, the answer came
                                   from the context and not from the memory; if
                                   it kept it after a delete, it is
                                   consolidation again, with a lucky answer that
                                   next time will not be

How a criterion is judged
-------------------------
Memora uses three LLM judges by majority. Not here: the criteria are written on
discrete values (2200, Torino, golden retriever) and text matching is enough,
costs zero calls and adds no variance to a measure whose purpose is precisely
to measure variance.

The case text matching does not catch on its own is the old value named as old:
*"non bevi piu' caffe'"* is the right answer to the question about coffee, but
the word "caffe'" is there. Hence the distinction between the two ways of
writing a forgetting criterion:

    forgets()     after a delete: the value must not appear, period.
                  The memory is gone, naming it is already a leak.
    supersedes()  after an update or a contradiction: the value may appear
                  only as superseded - in its own clause ("non piu' X",
                  "prima X"), or in a sentence that retracts it elsewhere or
                  that names the value that replaced it ("mangi pesce dopo
                  sei anni di vegetarianesimo").

The difference is not arbitrary, it is the semantics of the two operations: a
deletion must leave no trace, an update can be told as a change.

Two languages, not one
----------------------
The two columns read texts in different languages, and not by choice: the
answer follows the user's language and is Italian, the consolidated memories
follow the rule of the `fact` field and are English. So every criterion carries
the variants of both languages - "pesce" and "fish", "padre" and "father",
"corr*" and "run*" - and so do the superseding markers, which without English
would read "The user stopped drinking coffee" as a leak instead of a negation.

Putting them side by side instead of duplicating is mandatory: a criterion is a
fact, not a string. Splitting it into two criteria would double the denominator
of MPA and give 50% to a perfect but monolingual answer. In the right form the
operation goes the right way on both sides - on presence it widens, because any
variant is enough; on absence it tightens, because a value to forget is not
saved by translating itself.

Where the two languages share the root one entry is enough: "vegetarian*"
catches "vegetariana" and "vegetarian", "clinic*" catches "clinica" and
"clinic", "eliminat*" catches "eliminato" and "eliminated".

One distortion remains, and it is worth knowing: the superseding check looks at
the value's sentence, so a sentence that contradicts itself ("bevi caffe' la
mattina, ma hai eliminato il caffe'") passes, and the old value named in a
sentence of its own, without markers, is counted as an error.

Faithfulness to the reference
-----------------------------
The computation is that of `fama_score` and `overall_metrics` in
evals/model_eval/model_based_evaluator.py. They match: the per-question formula,
lam, the clip at zero, FAA = 1 when there are no absence criteria, the overall
score as the per-question mean multiplied by a hundred, and the overall MPA and
FAA as micro-averages over the criteria instead of per-question means.

Three things do not match, and they are all declared:

1. **The judge.** Memora uses GPT-4.1, Claude Haiku 4.5 and Gemini 2.5 Flash by
   majority on criteria in natural language. Here the match is textual on
   hand-written variants. Same formula, criteria judged differently.

2. **MPA on an empty set.** Their code sets it to 0, and since there lam is 1
   those questions are worth 0 whatever happens. In their dataset the branch
   never runs; here it would run nine times out of 56, capping the total below
   84 points regardless. It is 1 by vacuity, as the paper's prose says.
   The computation is in `score`, and so is the reason.

3. **The task types.** Memora splits into Remembering, Reasoning and
   Recommending and reports a score for each. Here almost all the questions are
   Remembering, and ten are Recommending-type tasks ("mi consigli cosa cucinare
   per cena?"), judged with the same criteria: what the answer and the memory
   must contain. The number stays a single one.

So the number is computed as in the paper, but **it is not comparable with its
tables**: different data, one person instead of ten, 56 questions instead of a
hundred and fifty, and another judge. It is a number to compare this
architecture with itself from one run to the next.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


# --------------------------------------------------------------------------- #
# Normalization and match
# --------------------------------------------------------------------------- #

# The conversation is written in ASCII with the apostrophe ("caffe'", "e'"), the
# model answers with real accents ("caffe'" -> "caffè"). Without normalizing,
# half of the criteria would catch nothing for a typographical reason.
_COMBINING = "Mn"
_SEPARATORS = ("'", "’", "ʼ", "`", "-", "‐", "–", "—")

# How wide the gap of a variant with the '*' in the middle can be:
# "non*piu" must catch "non fai piu'" without reaching the next sentence.
_GAP = 24

# What a clause breaks on. The comma is there on purpose: the superseding markers
# are searched in the value's clause, and without the comma
# "bevi caffe', non te'" would pass for a negation of the coffee.
_CLAUSE_BREAK = re.compile(r"[.!?;:,\n]")

# What a sentence breaks on: never past a line break, which in the memory
# separates two different facts.
_SENTENCE_BREAK = re.compile(r"[.!?\n]")

# The value appears, but negated: the memory was deleted or contradicted.
# The markers carry their own '*': an asterisk where the word family is needed
# ("eliminat*"), a whole word where the prefix would catch too much ("non"
# without asterisk, otherwise it catches "nonna"; English "no", otherwise it
# catches "not", "now", "north" and half the dictionary).
#
# They are in two languages because the two columns of the report read texts in
# different languages: the answer follows the user's language and is Italian,
# the consolidated memories follow the rule of the `fact` field and are English.
# With only the Italian markers, "The user stopped drinking coffee" would be
# counted as a leak instead of a negation, and the memory column would collapse
# for lexical reasons exactly where consolidation works.
_NEGATION = (
    # Italian
    "non", "mai", "nessun*", "niente", "nulla", "senza", "smesso", "smettere",
    "eliminat*", "tolt*", "rimoss*", "dimenticat*", "abbandonat*",
    "rinunciat*", "esclus*",
    # English ("cancel*" also covers "cancellato", "eliminat*" also
    # "eliminated": where the two languages share the root one entry is enough)
    "no", "not", "never", "none", "nothing", "without", "anymore", "longer",
    "stop*", "quit*", "remov*", "delet*", "cancel*", "forgot*", "forget*",
    "drop*", "abandon*", "exclud*",
    # contractions: the apostrophe becomes a space in `normalize`, so "doesn't"
    # arrives here as "doesn t" and the whole word to catch is "doesn"
    "doesn", "don", "didn", "isn", "wasn", "aren",
)

# The value appears, but as an old value: there has been a change.
# The present-tense markers - "ora", "adesso", "now" - are left out on purpose:
# they sit in the clause of the NEW value, not of the old one, and admitting them
# would let "ora vive a Mondello" pass as if it were a superseding.
_TRANSITION = (
    # Italian
    "piu", "prima", "era", "erano", "ero", "eri", "invece", "sostituit*",
    "cambiat*", "aggiornat*", "modificat*", "passat*", "precedent*",
    "trasferit*", "vecchi*", "fino", "allora", "inizialmente", "originariamente",
    # English
    "previously", "was", "were", "instead", "replac*", "chang*", "updat*",
    "moved", "relocat*", "former", "formerly", "old", "until", "used",
    "initially", "originally",
)

_SUPERSEDED_MARKERS = _NEGATION + _TRANSITION


def normalize(text) -> str:
    """Lowercase, without accents, apostrophes and hyphens, spaces collapsed.

    It makes the spellings going around in this test coincide: "caffe'" of the
    conversation, "caffè" of the answer, "Caffe" of a consolidated memory,
    "part-time" and "part time".

    Line breaks stay: in the memory they separate one fact from the other, and
    _CLAUSE_BREAK splits on them. Collapsing them like the other spaces merged
    the store into a single clause, and a "no longer" in any memory made every
    value of the archive pass as superseded.
    """
    lowered = unicodedata.normalize("NFD", str(text or "").lower())
    stripped = "".join(ch for ch in lowered if unicodedata.category(ch) != _COMBINING)
    for separator in _SEPARATORS:
        stripped = stripped.replace(separator, " ")
    lines = (re.sub(r"\s+", " ", line).strip() for line in stripped.split("\n"))
    return "\n".join(line for line in lines if line)


def _pattern(variant: str) -> str:
    """Regex of a variant. The '*' means two things, depending on where it is.

    At the end it is a prefix: "corr*" catches correre, corri, corro without
    having to list them. In the middle it is a gap: "non*piu" catches "non fai
    piu'" and "non lo suono piu'", which are the same phrase with words in
    between. Without the '*' the variant is a whole word.
    """
    variant = normalize(variant)
    open_end = variant.endswith("*")
    parts = [piece.strip() for piece in variant.split("*") if piece.strip()]
    if not parts:
        return r"(?!x)x"  # a variant made only of '*' catches nothing
    body = (r".{0," + str(_GAP) + r"}?").join(re.escape(piece) for piece in parts)
    return r"\b" + body + ("" if open_end else r"\b")


def _occurrences(haystack: str, variants: Sequence[str]) -> List[int]:
    """The positions where any of the variants appears in the text."""
    found = []
    for variant in variants:
        found.extend(match.start() for match in re.finditer(_pattern(variant), haystack))
    return sorted(found)


def _clause_around(text: str, position: int) -> str:
    """The clause that contains that position."""
    start = 0
    for match in _CLAUSE_BREAK.finditer(text, 0, position):
        start = match.end()
    end = _CLAUSE_BREAK.search(text, position)
    return text[start:end.start() if end else len(text)]


def _sentence_around(text: str, position: int) -> Tuple[int, int]:
    """Start and end of the sentence that contains that position."""
    start = 0
    for match in _SENTENCE_BREAK.finditer(text, 0, position):
        start = match.end()
    end = _SENTENCE_BREAK.search(text, position)
    return start, end.start() if end else len(text)


# A clause that follows the value and starts like this is correcting it...
_ADVERSATIVE = ("ma", "pero", "tuttavia", "eppure", "but", "however", "though", "yet")

# ...as long as it says it is over: a plain negation is not enough, "bevi caffe',
# ma non alcolici" negates something else.
_CESSATION = ("piu", "smesso", "smettere", "eliminat*", "tolt*", "abbandonat*",
              "rinunciat*", "anymore", "longer", "stop*", "quit*", "gave up")


def _has_marker(clause: str, markers: Sequence[str]) -> bool:
    return any(re.search(_pattern(marker), clause) for marker in markers)


def _following_clause(text: str, position: int) -> str:
    """The clause after the value's one, if it is in the same sentence.

    Never past a full stop or a line break: in the memory every line is a
    different fact.
    """
    end = _CLAUSE_BREAK.search(text, position)
    if end is None or end.group() not in ",;:":
        return ""
    after = _CLAUSE_BREAK.search(text, end.end())
    return text[end.end():after.start() if after else len(text)].strip()


def _is_superseded(text: str, position: int) -> bool:
    """Is the value in that position named as superseded or negated?

    In its own clause, or in the next one if that one corrects it:
    "bevi solo caffe' la mattina, ma ora non lo prendi piu'" says the coffee
    is over, with the negation after the comma and a pronoun instead of the noun.
    """
    if _has_marker(_clause_around(text, position), _SUPERSEDED_MARKERS):
        return True
    following = _following_clause(text, position)
    return (following.split(" ", 1)[0] in _ADVERSATIVE
            and _has_marker(following, _CESSATION))


# --------------------------------------------------------------------------- #
# The criteria
# --------------------------------------------------------------------------- #

PRESENCE = "memory_presence"
FORGETTING = "forgetting_absence"


@dataclass(frozen=True)
class Criterion:
    """A binary criterion, in the form of Memora's two `evaluation_type` values."""

    label: str
    variants: Tuple[str, ...]
    kind: str
    superseded_ok: bool = False
    # Only for recalls_some: the facts to choose from, each with its own
    # variants, and how many of them must appear.
    groups: Tuple[Tuple[str, ...], ...] = ()
    minimum: int = 1
    # Only for supersedes, and only where the new value excludes the old one (2200
    # instead of 2000): guitar and piano lived together, they do not replace each other.
    replaced_by: Tuple[str, ...] = ()

    def satisfied(self, text: str) -> bool:
        """Is the criterion met by this text?"""
        haystack = normalize(text)
        if self.kind == PRESENCE and self.groups:
            found = sum(1 for group in self.groups if _occurrences(haystack, group))
            return found >= self.minimum
        hits = _occurrences(haystack, self.variants)
        if self.kind == PRESENCE:
            return bool(hits)
        if not self.superseded_ok:
            return not hits
        # A mention not superseded in its clause passes if the same sentence
        # retracts the value elsewhere, or names the value that replaced it.
        retracted = [position for position in hits if _is_superseded(haystack, position)]
        narrated = retracted + _occurrences(haystack, self.replaced_by)
        for position in hits:
            if position in retracted:
                continue
            start, end = _sentence_around(haystack, position)
            if not any(start <= other < end for other in narrated):
                return False
        return True


def recalls(label: str, *variants: str) -> Criterion:
    """Presence criterion: this information must be there."""
    return Criterion(label, tuple(variants) or (label,), PRESENCE)


def recalls_some(label: str, minimum: int, *facts) -> Criterion:
    """Presence criterion for open questions: at least `minimum` of these facts.

    Each fact is a variant or a tuple of variants, and it counts only once even
    if it appears in two languages. generate_answer answers in one or two
    sentences: to "cosa sai della mia alimentazione?" it chooses which facts to
    say, and a criterion that demands the exact ones measures the choice, not
    the memory.
    """
    groups = tuple((fact,) if isinstance(fact, str) else tuple(fact) for fact in facts)
    variants = tuple(variant for group in groups for variant in group)
    return Criterion(label, variants, PRESENCE, groups=groups, minimum=minimum)


def forgets(label: str, *variants: str) -> Criterion:
    """Absence criterion after a delete: the value must not appear.

    Not even negated. The memory was deleted at the user's request: answering
    "non abiti piu' a Mondello" reveals exactly what had to disappear.
    """
    return Criterion(label, tuple(variants) or (label,), FORGETTING, superseded_ok=False)


def supersedes(label: str, *variants: str, replaced_by: Sequence[str] = ()) -> Criterion:
    """Absence criterion after an update or a contradiction.

    The old value may appear, but only as old: telling the change is legitimate,
    presenting the superseded value again as current is not.
    """
    return Criterion(label, tuple(variants) or (label,), FORGETTING, superseded_ok=True,
                     replaced_by=tuple(replaced_by))


@dataclass(frozen=True)
class EvaluationQuestion:
    """A question of the conversation with its criteria."""

    index: int
    note: str
    criteria: Tuple[Criterion, ...]

    @property
    def presence(self) -> List[Criterion]:
        """The presence criteria."""
        return [item for item in self.criteria if item.kind == PRESENCE]

    @property
    def forgetting(self) -> List[Criterion]:
        """The absence criteria."""
        return [item for item in self.criteria if item.kind == FORGETTING]


# --------------------------------------------------------------------------- #
# The score
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Score:
    """The outcome of a question: the three quantities, the counts and what failed."""

    mpa: float
    faa: float
    lam: float
    fama: float
    n_presence: int
    n_forgetting: int
    missed: Tuple[str, ...]
    leaked: Tuple[str, ...]

    @property
    def clean(self) -> bool:
        """No criterion violated."""
        return not self.missed and not self.leaked

    @property
    def presence_satisfied(self) -> int:
        """How many presence criteria were satisfied."""
        return self.n_presence - len(self.missed)

    @property
    def forgetting_satisfied(self) -> int:
        """How many absence criteria were satisfied."""
        return self.n_forgetting - len(self.leaked)


def score(question: EvaluationQuestion, text: str) -> Score:
    """FAMA of a question against a text: the answer, or the active memories.

    It follows `fama_score` of evals/model_eval/model_based_evaluator.py, with a
    declared deviation on the case without presence criteria - see below.
    """
    presence, forgetting = question.presence, question.forgetting
    n_p, n_f = len(presence), len(forgetting)

    missed = tuple(item.label for item in presence if not item.satisfied(text))
    leaked = tuple(item.label for item in forgetting if not item.satisfied(text))

    # DECLARED DEVIATION. Memora's code sets MPA to 0 when there are no presence
    # criteria, and since there lam is 1, those questions are worth 0 whatever
    # happens. In their dataset it never happens - the questions are lists
    # ("quali task restano?") and something always remains - so it is a
    # defensive branch that never runs. Here instead it happens nine times out
    # of 56: after a deletion there is no valid value to remember, and "non lo
    # so" is the right answer. With MPA at 0 those nine questions could never be
    # right and the total would be capped below 84 points, whatever the
    # architecture does.
    #
    # Here MPA is 1 by vacuity, which is the reading of the paper's prose:
    # "the fraction of memory presence criteria satisfied" on an empty set,
    # with lam = 1 handing all the weight to FAA. FAMA degenerates into FAA, and
    # that is exactly what those questions must measure.
    mpa = 1.0 - len(missed) / n_p if n_p else 1.0
    faa = 1.0 - len(leaked) / n_f if n_f else 1.0
    lam = n_f / (n_p + n_f)

    return Score(mpa, faa, lam, max(0.0, mpa - lam * (1.0 - faa)),
                 n_p, n_f, missed, leaked)


def aggregate(scores: Sequence[Score]) -> Dict[str, float]:
    """The score of the architecture, not of the single question.

    It follows the reference's `overall_metrics`, and the three quantities are
    not all averaged the same way:

        fama      mean of the per-question FAMA, times a hundred. Every question
                  weighs the same, which is the meaning of an already normalized
                  score.
        mpa, faa  ratio between satisfied criteria and total criteria over all
                  the questions together (micro-average): there what weighs is
                  the criterion, not the question, so a question with seven
                  criteria counts for seven and not for one.
        accuracy  the same criteria, presence and absence in a single pile.
                  It is the reference's `overall_accuracy`: the raw fraction of
                  passed criteria, without the weight lam.

    Averaging mpa and faa per question too would give different and kinder
    numbers: a question with a single wrong criterion would weigh as much as one
    with seven right criteria.
    """
    presence_total = sum(item.n_presence for item in scores)
    forgetting_total = sum(item.n_forgetting for item in scores)
    presence_ok = sum(item.presence_satisfied for item in scores)
    forgetting_ok = sum(item.forgetting_satisfied for item in scores)
    criteria = presence_total + forgetting_total

    return {
        "questions": len(scores),
        "criteria": criteria,
        "fama": 100.0 * sum(item.fama for item in scores) / len(scores),
        "mpa": 100.0 * presence_ok / presence_total if presence_total else None,
        "faa": 100.0 * forgetting_ok / forgetting_total if forgetting_total else None,
        "accuracy": 100.0 * (presence_ok + forgetting_ok) / criteria if criteria else 0.0,
    }


def verdict(memory: Score, answer: Score, tolerance: float = 0.01) -> str:
    """Whose fault it is, reading the score on the two objects together.

    It is the quadrant of the table in the module docstring, translated into a
    single word to put at the end of the report line.

    The "memory wrong, answer right" quadrant though is not just one, and
    confusing it costs a diagnosis: if the memory **missed** a fact and the
    answer said it anyway, the answer came from the conversation's context and
    not from the memory; if instead the memory **kept** a deleted fact and the
    answer did not use it, the store is wrong and the answer was lucky - and
    that is consolidation, not luck to celebrate. The next question on the same
    fact loses it.
    """
    memory_ok = memory.fama >= 1.0 - tolerance
    answer_ok = answer.fama >= 1.0 - tolerance
    if memory_ok and answer_ok:
        return "ok"
    if memory_ok:
        return "answer"
    if not answer_ok:
        return "consolidation"
    return "consolidation" if memory.leaked else "outside memory"


def memory_snapshot_turn(question: int, maximum_historical_messages: int) -> int:
    """After the insert of which turn the memory is snapshotted for this question.

    Not at the moment of the question. Consolidation lags by construction:
    summarize_memories_node keeps the last maximum_historical_messages messages
    and consolidates the ones before, and every turn adds two (the question and
    the answer of generate_answer). The message right before the question -
    almost always the update or the delete the question puts to the test - at
    the moment of the question is still only in the conversation, and the memory
    "misses" it by definition. Measuring it there measured the lag, not
    consolidation.

    With the long test's window (2 messages) it is the insert of the question's
    own turn, which consolidates the message before and not yet the question.
    """
    return question + max(1, maximum_historical_messages // 2) - 1


# --------------------------------------------------------------------------- #
# The 56 questions
# --------------------------------------------------------------------------- #

QUESTIONS: Dict[int, EvaluationQuestion] = {

    36: EvaluationQuestion(
        36, "the facts of the presentation, nothing is superseded yet",
        (
            recalls_some(
                "at least two facts about the diet", 2,
                "vegetarian*", ("arachid*", "peanut*"), "2000", "protein*",
                ("alcol*", "alcohol*"), ("caffe", "coffee"), ("te verde", "green tea"),
                ("parmigian*", "parmesan"), ("liquirizia", "licorice", "liquorice"),
                ("glutine", "gluten", "celiac*"), ("lattosio", "lactose"),
            ),
        ),
    ),

    39: EvaluationQuestion(
        39, "msg 38 updated the calories from 2000 to 2200",
        (
            recalls("2200", "2200"),
            supersedes("2000", "2000", replaced_by=("2200",)),
        ),
    ),

    42: EvaluationQuestion(
        42, "msg 41 repeated the asthma of msg 9",
        (
            recalls("asthma", "asma", "asthma*"),
        ),
    ),

    46: EvaluationQuestion(
        46, "msg 45 repeated the name",
        (
            recalls("bianca"),
        ),
    ),

    50: EvaluationQuestion(
        50, "task: a film, and the horror films of msg 49 make the user anxious",
        (
            recalls("no horror", "horror*", "orror*"),
        ),
    ),

    58: EvaluationQuestion(
        58, "msg 57 contradicted the vegetarian diet: the user now eats fish",
        (
            recalls("eats fish", "pesce", "pescetarian*", "fish", "pescatarian*"),
            supersedes("vegetarian*",
                       replaced_by=("pesce", "fish", "pescatarian*", "pescetarian*")),
        ),
    ),

    60: EvaluationQuestion(
        60, "task: a dinner, with the fish of msg 57 and the peanut allergy",
        (
            recalls("eats fish", "pesce", "fish", "pescatarian*", "pescetarian*"),
            recalls("no peanuts", "arachid*", "peanut*"),
            supersedes("vegetarian*",
                       replaced_by=("pesce", "fish", "pescatarian*", "pescetarian*")),
        ),
    ),

    65: EvaluationQuestion(
        65, "msg 64 deleted the address: Palermo centre, then Mondello",
        (
            forgets("deleted address", "mondello", "palermo",
                    "centro", "centre", "center", "downtown"),
        ),
    ),

    67: EvaluationQuestion(
        67, "msg 66 raised the levothyroxine to 88 micrograms",
        (
            recalls("88 micrograms", "88"),
        ),
    ),

    71: EvaluationQuestion(
        71, "msg 70 dropped the coffee: no new drink in its place",
        (
            supersedes("morning coffee", "caffe", "coffee"),
        ),
    ),

    73: EvaluationQuestion(
        73, "msg 72: also allergic to penicillin",
        (
            recalls("penicillin", "penicillin*"),
        ),
    ),

    79: EvaluationQuestion(
        79, "msg 78: in August the user would like to go to Greece, maybe Naxos",
        (
            recalls("Naxos or Greece", "naxos", "grecia", "greece"),
        ),
    ),

    81: EvaluationQuestion(
        81, "task: getting to Naxos with the fear of flying of msg 75",
        (
            recalls("no plane", "volare", "fly*", "flight*", "aere*", "plane*",
                    "nave", "traghett*", "ferry*", "boat*"),
        ),
    ),

    84: EvaluationQuestion(
        84, "msg 83 deleted the age: first 29, then 30",
        (
            # Never the bare number: "6:30" becomes "6 30", and the alarm passed for the age.
            forgets("deleted age", "30 anni", "29 anni", "30 year*", "29 year*",
                    "is 30", "is 29", "turned 30", "turned 29", "aged 30", "age of 30",
                    "trent anni", "trentanni", "ventinove anni",
                    "thirty year*", "twenty nine year*"),
        ),
    ),

    90: EvaluationQuestion(
        90, "msg 89 moved the sister from Milano to Torino",
        (
            recalls("lives in Torino", "torino", "turin"),
            supersedes("old city Milano", "milano", "milan",
                       replaced_by=("torino", "turin")),
        ),
    ),

    92: EvaluationQuestion(
        92, "msg 85 and 91: the photography course on Monday evening",
        (
            recalls("photography", "fotograf*", "photograph*"),
        ),
    ),

    101: EvaluationQuestion(
        101, "msg 95: Giulia gets married on 14 June",
        (
            recalls("14 June", "14 giugno", "14 june", "june 14*", "14th of june",
                    "14th june"),
        ),
    ),

    105: EvaluationQuestion(
        105, "msg 104 deleted the sister's name",
        (
            forgets("deleted name", "chiara"),
        ),
    ),

    108: EvaluationQuestion(
        108, "msg 102: volunteering at a dog shelter on Tuesday evening",
        (
            recalls("volunteering", "volontari*", "volunteer*", "canile", "shelter*"),
        ),
    ),

    111: EvaluationQuestion(
        111, "msg 110 extended the allergy to peanut oil, msg 72 added penicillin"
             "penicillina",
        (
            recalls("peanuts", "arachid*", "peanut*"),
            recalls("penicillin", "penicillin*"),
        ),
    ),

    114: EvaluationQuestion(
        114, "msg 99: a change of mind about liquorice, now the user likes it",
        (
            recalls("liquorice", "liquirizia", "licorice", "liquorice"),
            recalls("likes it", "piace", "like*", "enjoy*", "ricredut*"),
        ),
    ),

    118: EvaluationQuestion(
        118, "msg 117 contradicted the breed: not a labrador but a golden retriever",
        (
            recalls("golden retriever", "golden"),
            supersedes("old breed labrador", "labrador", replaced_by=("golden",)),
        ),
    ),

    119: EvaluationQuestion(
        119, "msg 18: Argo cannot eat chicken",
        (
            recalls("chicken", "pollo", "chicken"),
        ),
    ),

    123: EvaluationQuestion(
        123, "task: the backpack for the mountains, with the asthma of msg 9",
        (
            recalls("inhaler", "inalator*", "inhaler*", "asma", "asthma*"),
        ),
    ),

    127: EvaluationQuestion(
        127, "msg 126 contradicted the yoga: stopped in September",
        (
            recalls("stopped", "smesso", "settembre", "non*piu", "interrott*",
                    "abbandonat*", "stop*", "quit*", "september", "no longer",
                    "anymore", "gave up"),
            supersedes("yoga days", "sabato", "mercoledi",
                       "saturday", "wednesday"),
        ),
    ),

    131: EvaluationQuestion(
        131, "task: the Thursday shopping, with the budget of msg 130",
        (
            recalls("50 euro budget", "50 eur*"),
        ),
    ),

    133: EvaluationQuestion(
        133, "msg 120: Argo's vaccine booster on 20 October",
        (
            recalls("20 October", "20 ottobre", "20 october", "october 20*",
                    "20th of october", "20th october"),
        ),
    ),

    137: EvaluationQuestion(
        137, "msg 128: dentist next Tuesday at three in the afternoon",
        (
            recalls("Tuesday", "martedi", "tuesday"),
            recalls("at three", "alle tre", "tre del pomeriggio", "15", "3 pm", "3pm",
                    "three in the afternoon", "3 in the afternoon"),
        ),
    ),

    140: EvaluationQuestion(
        140, "msg 139 deleted the job: clinic, then part-time, then self-employed",
        (

            forgets("deleted job", "fisioterapist*", "physiotherap*",
                    "physical therap*", "clinic*", "in proprio", "part time",
                    "self employed", "own practice", "freelance*"),
        ),
    ),

    148: EvaluationQuestion(
        148, "msg 147 quit the piano: guitar and jazz vinyls remain",
        (
            recalls("guitar", "chitarra", "guitar*"),
            recalls("jazz on vinyl", "jazz", "vinil*", "vinyl*"),
            supersedes("abandoned piano", "pianoforte", "piano"),
        ),
    ),

    151: EvaluationQuestion(
        151, "task: whom to leave Argo with, from msg 142",
        (
            recalls("the neighbour", "vicin*", "neighbo*"),
        ),
    ),

    154: EvaluationQuestion(
        154, "msg 153 set the calories to 1800, after 2000 and 2200",
        (
            recalls("goal 1800", "1800"),
            supersedes("goal 2200", "2200", replaced_by=("1800",)),
            supersedes("goal 2000", "2000", replaced_by=("1800",)),
        ),
    ),

    157: EvaluationQuestion(
        157, "msg 156 raised the levothyroxine to 100 micrograms, after the 88 of msg 66",
        (
            recalls("levothyroxine", "levotiroxin*", "levothyroxin*", "tiroide",
                    "thyroid*"),
            recalls("100 micrograms", "100"),
            supersedes("88 micrograms", "88", replaced_by=("100",)),
        ),
    ),

    169: EvaluationQuestion(
        169, "msg 168 deleted the coeliac disease: first the mother, then the father",
        (
            forgets("deleted coeliac relative", "padre", "madre", "papa",
                    "mamma", "father", "mother", "dad", "mom", "mum", "parent*"),
        ),
    ),

    171: EvaluationQuestion(
        171, "msg 25: the best friend is Giulia",
        (
            recalls("giulia"),
        ),
    ),

    172: EvaluationQuestion(
        172, "msg 53 and 158: Spanish, now at level B1",
        (
            recalls("Spanish", "spagnol*", "spanish"),
        ),
    ),

    177: EvaluationQuestion(
        177, "msg 176 took up running again; swimming is three times, yoga is over",
        (
            recalls("running taken up again", "corr*", "corsa", "run*", "jog*"),
            recalls("swimming", "nuot*", "swim*"),
            supersedes("yoga stopped", "yoga"),
        ),
    ),

    179: EvaluationQuestion(
        179, "task: the sports week, with the pilates of msg 178 and the running of msg 176"
             "msg 176",
        (
            recalls_some("at least two sports", 2, ("corr*", "corsa", "run*", "jog*"),
                         ("nuot*", "swim*"), "pilates"),
            supersedes("yoga stopped", "yoga"),
        ),
    ),

    181: EvaluationQuestion(
        181, "msg 180: chili peppers join the tomatoes and basil of msg 31",
        (
            recalls_some("at least two plants", 2, ("pomodor*", "tomato*"),
                         ("basilic*", "basil"), ("peperoncin*", "chili*", "chilli*")),
        ),
    ),

    183: EvaluationQuestion(
        183, "msg 182: stopped meditating",
        (
            recalls("meditation", "medit*"),
            recalls("stopped", "smesso", "stop*", "no longer", "non*piu", "anymore",
                    "quit*", "gave up"),
        ),
    ),

    188: EvaluationQuestion(
        188, "msg 187 deleted the volunteering at the dog shelter",
        (
            forgets("deleted volunteering", "volontari*", "volunteer*", "canile",
                    "shelter*", "kennel*"),
        ),
    ),

    192: EvaluationQuestion(
        192, "task: the lake without a driving licence (msg 6 and 191)",
        (
            recalls("no driving licence", "patente", "licen*"),
        ),
    ),

    194: EvaluationQuestion(
        194, "msg 193: got over the fear of flying",
        (
            recalls("flying", "volare", "fly*", "flight*", "aere*", "plane*"),
            recalls("got over", "superat*", "overc*", "got over"),
        ),
    ),

    198: EvaluationQuestion(
        198, "msg 197 deleted the relatives, not the animals",
        (
            recalls("the dog Argo", "argo"),
            recalls("the cat Milo", "milo"),
        ),
    ),

    200: EvaluationQuestion(
        200, "msg 199: eye surgery, no more glasses",
        (
            recalls("eye surgery", "operat*", "surgery", "intervent*", "laser"),
            supersedes("glasses", "occhiali", "glasses", "eyeglasses"),
        ),
    ),

    201: EvaluationQuestion(
        201, "msg 3: the birthday is 12 March",
        (
            recalls("12 March", "12 marzo", "12 march", "march 12*", "12th of march",
                    "12th march"),
        ),
    ),

    203: EvaluationQuestion(
        203, "msg 202 deleted the birthday date",
        (
            forgets("deleted birthday", "12 marzo", "12 march", "march 12*",
                    "marzo", "march"),
        ),
    ),

    206: EvaluationQuestion(
        206, "msg 205: finished the sign language course",
        (
            recalls("sign language", "lingua dei segni", "segni", "sign language"),
            recalls("finished", "finit*", "conclus*", "finished", "completed",
                    "primo livello", "first level"),
        ),
    ),

    208: EvaluationQuestion(
        208, "msg 51 raised the proteins from 60 to 75 grams",
        (
            recalls("75 grams", "75"),
            supersedes("60 grams", "60", replaced_by=("75",)),
        ),
    ),

    210: EvaluationQuestion(
        210, "msg 209 confirmed the two litres of water of msg 68",
        (
            recalls("two litres", "due litri", "2 litri", "two lit*", "2 lit*", "2l"),
        ),
    ),

    213: EvaluationQuestion(
        213, "msg 212 deleted the vegetable garden on the balcony",
        (
            forgets("deleted vegetable garden", "orto", "pomodor*", "tomato*", "basilic*",
                    "basil", "peperoncin*", "chili*", "chilli*", "vegetable garden",
                    "vegetable patch"),
        ),
    ),

    217: EvaluationQuestion(
        217, "msg 216: no Naxos, Portugal in August",
        (
            recalls("Portugal", "portogallo", "portugal"),
            supersedes("Naxos and Greece", "naxos", "grecia", "greece",
                       replaced_by=("portogallo", "portugal")),
        ),
    ),

    219: EvaluationQuestion(
        219, "task: a dinner, with the favourite trattoria of msg 218",
        (
            recalls("the trattoria in Cefalu'", "cefalu", "trattoria"),
        ),
    ),

    230: EvaluationQuestion(
        230, "msg 229: no skiing in Trentino, at Christmas the user stays home",
        (
            recalls("stays home", "resto a casa", "resti a casa", "restare a casa",
                    "rimani a casa", "stay at home", "stays at home", "staying at home",
                    "stay home", "stays home", "staying home"),
            supersedes("skiing in Trentino", "sciare", "sci", "ski", "skiing", "trentino"),
        ),
    ),

    231: EvaluationQuestion(
        231, "msg 161: the favourite writer is Camilleri",
        (
            recalls("camilleri"),
        ),
    ),

    232: EvaluationQuestion(
        232, "the overall picture: what survived and what had to disappear",
        (
            recalls_some(
                "at least two surviving facts", 2,
                "bianca", ("arachid*", "peanut*"), "argo", "milo",
                ("pesce", "fish", "pescatarian*", "pescetarian*"), "protein*",
                ("te verde", "green tea"), ("nuot*", "swim*"), ("parmigian*", "parmesan"),
                ("chitarra", "guitar*"), ("jazz", "vinil*", "vinyl*"),
                ("grecia", "greece", "naxos"), ("montagna", "mountain*"),
                ("acqua", "water"), ("lingua dei segni", "sign language"),
                ("bici*", "bike*", "bicycle*"),
            ),
            forgets("the sister's name", "chiara"),
            forgets("job", "fisioterapist*", "physiotherap*",
                    "physical therap*", "clinic*"),
            forgets("address", "mondello", "palermo"),
        ),
    ),
}

