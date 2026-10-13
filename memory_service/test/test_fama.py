"""The FAMA computation, checked without touching the model.

The long test costs a long run and is not repeatable: if the metric
that judges it has a bug, the bug is found at the end and paid for again. Here
instead the formula, the match and the criteria table are checked in half a
second, against the paper's example and against the cases that match would
happily get wrong - the old value named as old, and the old value named after
a negation that was about something else.

Run: python memory_service/run_tests.py -v
"""

import ast
import os
import sys
import unittest

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (PACKAGE_ROOT, os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

import fama  # noqa: E402


LONGRUN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "test_long_term_interaction.py")

# The questions asked right after a deletion. The value must not appear even
# negated: the memory is gone, naming it is already a leak.
AFTER_DELETE = (65, 84, 105, 140, 169, 188, 203, 213)


def conversation():
    """CONVERSATION read from the source, without importing the module.

    Importing it would pull in langgraph and the real stack just to read a list
    of strings, and this test must run also where they are missing.
    """
    with open(LONGRUN, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "CONVERSATION":
                return ast.literal_eval(node.value)
    raise AssertionError("CONVERSATION not found in test_long_term_interaction.py")


def question(*criteria):
    """A stand-in question, to try the formula without the real table."""
    return fama.EvaluationQuestion(0, "prova", tuple(criteria))


class FormulaTest(unittest.TestCase):
    """FAMA = max(0, MPA - lam * (1 - FAA)), with lam = N_forget / N_total."""

    def test_the_paper_example_weights_forgetting_by_its_share(self):
        """5 presence criteria and 8 absence ones: lam = 8/13, as in Memora.

        It is question activity_todos_158 of the academic_researcher persona,
        the only one reported in full in the repo: five tasks that must be
        there, eight deleted ones that must not.
        """
        subject = question(
            *[fama.recalls("task %d" % n, "task%d" % n) for n in range(5)],
            *[fama.forgets("vecchio %d" % n, "vecchio%d" % n) for n in range(8)],
        )

        perfect = fama.score(subject, "task0 task1 task2 task3 task4")
        self.assertEqual(perfect.lam, 8 / 13)
        self.assertEqual(perfect.fama, 1.0)

        # All the memory, no forgetting: the penalty is exactly lam.
        leaking = fama.score(subject, "task0 task1 task2 task3 task4 "
                                      "vecchio0 vecchio1 vecchio2 vecchio3 "
                                      "vecchio4 vecchio5 vecchio6 vecchio7")
        self.assertEqual(leaking.mpa, 1.0)
        self.assertEqual(leaking.faa, 0.0)
        self.assertAlmostEqual(leaking.fama, 1 - 8 / 13)

    def test_without_forgetting_criteria_fama_is_just_recall(self):
        subject = question(fama.recalls("torino"), fama.recalls("argo"))
        scored = fama.score(subject, "vive a Torino")
        self.assertEqual(scored.lam, 0.0)
        self.assertEqual(scored.fama, scored.mpa)
        self.assertEqual(scored.fama, 0.5)

    def test_without_presence_criteria_fama_is_just_forgetting(self):
        """The questions after a delete: nothing to say, only to keep quiet about."""
        subject = question(fama.forgets("indirizzo", "mondello"))
        self.assertEqual(fama.score(subject, "Non lo so.").fama, 1.0)

        leaked = fama.score(subject, "Abiti a Mondello.")
        self.assertEqual(leaked.lam, 1.0)
        self.assertEqual(leaked.faa, 0.0)
        self.assertEqual(leaked.fama, 0.0)

    def test_the_score_never_goes_below_zero(self):
        subject = question(fama.recalls("torino"),
                           fama.forgets("milano"), fama.forgets("palermo"))
        scored = fama.score(subject, "Vivi a Milano, o forse a Palermo.")
        self.assertEqual(scored.mpa, 0.0)
        self.assertEqual(scored.faa, 0.0)
        self.assertEqual(scored.fama, 0.0)

    def test_the_score_says_which_criteria_gave_way(self):
        subject = question(fama.recalls("torino"), fama.forgets("milano"))
        scored = fama.score(subject, "Vive a Milano.")
        self.assertEqual(scored.missed, ("torino",))
        self.assertEqual(scored.leaked, ("milano",))
        self.assertFalse(scored.clean)

    def test_the_overall_fama_is_the_mean_over_questions(self):
        scores = [fama.score(question(fama.recalls("a")), "a"),
                  fama.score(question(fama.recalls("a")), "b")]
        overall = fama.aggregate(scores)
        self.assertEqual(overall["fama"], 50.0)
        self.assertEqual(overall["questions"], 2)
        self.assertEqual(overall["criteria"], 2)

    def test_the_overall_mpa_and_faa_are_micro_averages(self):
        """On the criteria, not on the questions: one with seven criteria weighs seven.

        It is how the reference's `overall_metrics` computes them. With the
        per-question mean the same two outcomes would give 66.7 instead of 50.0,
        and a question with a single wrong criterion would weigh as much as one
        with seven right criteria.
        """
        scores = [
            fama.score(question(fama.recalls("a")), "a"),
            fama.score(question(fama.recalls("a"), fama.recalls("b"),
                                fama.recalls("c")), "a"),
        ]
        overall = fama.aggregate(scores)
        self.assertEqual(overall["mpa"], 50.0)          # 2 criteria passed out of 4
        self.assertIsNone(overall["faa"])               # no absence criterion
        self.assertAlmostEqual(overall["fama"], 100.0 * (1 + 1 / 3) / 2)

    def test_the_overall_accuracy_pools_both_families(self):
        """The reference's `overall_accuracy`: passed criteria over the total."""
        scores = [fama.score(question(fama.recalls("a"), fama.forgets("b")), "a b")]
        overall = fama.aggregate(scores)
        self.assertEqual(overall["mpa"], 100.0)
        self.assertEqual(overall["faa"], 0.0)
        self.assertEqual(overall["accuracy"], 50.0)     # 1 criterion out of 2
        self.assertEqual(overall["fama"], 50.0)


class MatchingTest(unittest.TestCase):
    """How a value is recognized as appearing in a text."""

    def test_accents_apostrophes_and_hyphens_are_the_same_word(self):
        """The spellings of caffe' must coincide, or half the criteria are blind."""
        self.assertEqual(fama.normalize("Caffè"), "caffe")
        self.assertEqual(fama.normalize("caffe'"), "caffe")
        self.assertEqual(fama.normalize("part-time"), "part time")

        criterion = fama.recalls("caffe", "caffe")
        self.assertTrue(criterion.satisfied("Bevi il caffè la mattina."))
        self.assertTrue(criterion.satisfied("bevi il caffe' la mattina"))

    def test_line_breaks_survive_normalization(self):
        self.assertEqual(fama.normalize("Caffè  la\n\nmattina \n tè"), "caffe la\nmattina\nte")

    def test_a_trailing_star_matches_the_whole_family(self):
        criterion = fama.recalls("corsa", "corr*")
        self.assertTrue(criterion.satisfied("Corri tre volte a settimana."))
        self.assertTrue(criterion.satisfied("Hai ripreso a correre."))
        self.assertFalse(criterion.satisfied("Fai nuoto."))

    def test_a_star_in_the_middle_bridges_the_words_between(self):
        """'non*piu' must catch 'non fai piu'', which is the same phrase."""
        criterion = fama.recalls("ha smesso", "non*piu")
        self.assertTrue(criterion.satisfied("No, non fai piu' yoga."))
        self.assertTrue(criterion.satisfied("Non lo suoni più."))
        self.assertFalse(criterion.satisfied(
            "Non fai yoga il sabato e vai a correre molto presto la "
            "mattina, come sempre, quindi dormi di piu'."))

    def test_a_word_variant_does_not_match_inside_another_word(self):
        self.assertFalse(fama.recalls("centro", "centro")
                         .satisfied("Mi concentro sul lavoro."))
        self.assertFalse(fama.recalls("piano", "pianoforte")
                         .satisfied("Suoni il piano."))

    def test_presence_is_satisfied_by_any_variant(self):
        criterion = fama.recalls("jazz in vinile", "jazz", "vinil*")
        self.assertTrue(criterion.satisfied("Collezioni vinili."))
        self.assertTrue(criterion.satisfied("Ascolti jazz."))
        self.assertFalse(criterion.satisfied("Suoni la chitarra."))

    def test_empty_text_misses_presence_and_satisfies_absence(self):
        self.assertFalse(fama.recalls("torino").satisfied(""))
        self.assertTrue(fama.forgets("milano").satisfied(""))


class DeleteVersusUpdateTest(unittest.TestCase):
    """Deleting and superseding are not the same thing, and the criteria say so."""

    def test_after_a_delete_the_value_must_not_appear_even_negated(self):
        """Saying 'non abiti piu' a Mondello' reveals what had to disappear."""
        criterion = fama.forgets("indirizzo", "mondello")
        self.assertFalse(criterion.satisfied("Non abiti più a Mondello."))
        self.assertFalse(criterion.satisfied("Abiti a Mondello."))
        self.assertTrue(criterion.satisfied("Non ho questa informazione."))

    def test_after_an_update_the_old_value_may_appear_as_old(self):
        criterion = fama.supersedes("vecchio obiettivo", "2000")
        self.assertTrue(criterion.satisfied("L'obiettivo è 2200, non più 2000."))
        self.assertTrue(criterion.satisfied("Era 2000, ora è 2200."))
        self.assertFalse(criterion.satisfied("Il tuo obiettivo è 2000 calorie."))

    def test_a_move_counts_as_superseding_the_old_city(self):
        criterion = fama.supersedes("vecchia citta'", "milano")
        self.assertTrue(criterion.satisfied("Vive a Torino, si è trasferita da Milano."))
        self.assertFalse(criterion.satisfied("Tua sorella vive a Milano."))

    def test_a_negation_does_not_leak_across_the_clause(self):
        """The negation holds in its own clause, not in the whole answer.

        Without this, "Non bevi alcolici. Bevi caffe' la mattina." would pass:
        the word "non" is there, but it negates something else.
        """
        criterion = fama.supersedes("caffe'", "caffe")
        self.assertFalse(criterion.satisfied("Non bevi alcolici. Bevi caffè la mattina."))
        self.assertFalse(criterion.satisfied("Non fai più yoga, e bevi caffè la mattina."))
        self.assertTrue(criterion.satisfied("Non bevi più caffè."))

    def test_english_negations_supersede_too(self):
        """The memories are English: without these markers they would all be leaks."""
        criterion = fama.supersedes("caffe'", "caffe", "coffee")
        for memory in ("The user stopped drinking coffee.",
                       "The user no longer drinks coffee.",
                       "The user does not drink coffee anymore.",
                       "The user removed coffee from the morning routine.",
                       "The user used to drink coffee in the morning."):
            self.assertTrue(criterion.satisfied(memory), memory)
        self.assertFalse(criterion.satisfied("The user drinks coffee in the morning."))

    def test_the_english_negation_also_stays_in_its_clause(self):
        criterion = fama.supersedes("caffe'", "caffe", "coffee")
        self.assertFalse(criterion.satisfied(
            "The user does not drink alcohol. The user drinks coffee."))

    def test_the_bare_italian_no_does_not_cover_the_sentence_after_it(self):
        """The only English marker that collides with Italian is "no".

        In English it opens "no longer"; in Italian it is the curt answer, and
        without the cut on punctuation it would cover the whole sentence that
        follows - "No, tua sorella vive a Milano" would pass for a superseding.

        A known gap remains: "No tua sorella vive a Milano" without a comma is a
        single clause and passes. It takes an ungrammatical answer to hit it, and
        closing the leak would mean removing "no", which all the English
        memories need.
        """
        criterion = fama.supersedes("vecchia citta'", "milano", "milan")
        self.assertFalse(criterion.satisfied("No, tua sorella vive a Milano."))
        self.assertFalse(criterion.satisfied("No. Tua sorella vive a Milano."))
        self.assertTrue(criterion.satisfied("No, si è trasferita a Torino da Milano."))

    def test_the_present_tense_is_not_a_marker_in_either_language(self):
        """"ora" and "now" sit in the sentence of the NEW value, not of the old one."""
        criterion = fama.supersedes("indirizzo", "mondello")
        self.assertFalse(criterion.satisfied("The user now lives in Mondello."))
        self.assertFalse(criterion.satisfied("Adesso abiti a Mondello."))

    def test_a_negation_in_one_memory_does_not_cover_another(self):
        """Every memory is a line, and every line a clause of its own.

        Before, line breaks became spaces and the store a single clause: the
        piano's "no longer" covered the still active coffee, and FAMA-memory
        gave 1.00 to an archive that had forgotten nothing.
        """
        criterion = fama.supersedes("caffe'", "caffe", "coffee")
        self.assertFalse(criterion.satisfied(
            "The user no longer plays the piano\nThe user drinks coffee every morning"))

    def test_a_correction_after_but_supersedes_what_came_before(self):
        criterion = fama.supersedes("caffe'", "caffe", "coffee")
        self.assertTrue(criterion.satisfied(
            "Di solito bevi solo caffè la mattina, ma ora non lo prendi più."))
        self.assertTrue(criterion.satisfied(
            "The user drank coffee in the morning, but not anymore."))

    def test_a_but_that_negates_something_else_does_not(self):
        criterion = fama.supersedes("caffe'", "caffe", "coffee")
        self.assertFalse(criterion.satisfied("Bevi caffè la mattina, ma non alcolici."))
        self.assertFalse(criterion.satisfied("Bevi caffè la mattina, e non più tè."))
        self.assertFalse(criterion.satisfied("Bevi caffè la mattina. Ma non più tè."))

    def test_all_occurrences_must_be_acceptable_not_just_one(self):
        criterion = fama.supersedes("caffe'", "caffe")
        self.assertFalse(criterion.satisfied(
            "Non bevi più caffè. Il caffè lo prendi alle otto."))

    def test_a_retraction_covers_the_whole_sentence(self):
        criterion = fama.supersedes("caffe'", "caffe", "coffee")
        self.assertTrue(criterion.satisfied(
            "La memoria diceva che bevevi caffè; poi hai eliminato il caffè."))
        self.assertFalse(criterion.satisfied(
            "Bevevi caffè la mattina. Poi hai eliminato il caffè del pomeriggio."))
        self.assertFalse(criterion.satisfied(
            "The user no longer drinks coffee\nThe user drinks coffee every morning"))

    def test_the_replacing_value_in_the_same_sentence_narrates_the_change(self):
        criterion = fama.supersedes("milano", "milano", replaced_by=("torino",))
        self.assertTrue(criterion.satisfied("Vive a Torino, ci è andata da Milano."))
        self.assertFalse(criterion.satisfied("Vive a Milano. A Torino va in vacanza."))
        self.assertFalse(fama.supersedes("milano", "milano").satisfied(
            "Vive a Torino, ci è andata da Milano."))


class OpenQuestionTest(unittest.TestCase):
    """Open questions want some facts, not those facts."""

    def setUp(self):
        self.criterion = fama.recalls_some(
            "alimentazione", 2,
            "vegetarian*", ("arachid*", "peanut*"), "glutine", "lattosio")

    def test_any_two_facts_are_enough(self):
        self.assertTrue(self.criterion.satisfied(
            "Cuoci senza glutine e sei intollerante al lattosio."))
        self.assertTrue(self.criterion.satisfied("Sei vegetariana e allergica alle arachidi."))

    def test_one_fact_is_not(self):
        self.assertFalse(self.criterion.satisfied("Sei allergica alle arachidi."))

    def test_one_fact_in_two_languages_counts_once(self):
        self.assertFalse(self.criterion.satisfied("Allergica alle arachidi (peanuts)."))


class SnapshotTimingTest(unittest.TestCase):
    """The memory is measured when the message before the question is consolidated."""

    def test_with_the_long_run_window_it_is_the_question_turn(self):
        self.assertEqual(fama.memory_snapshot_turn(45, 2), 45)

    def test_a_wider_window_pushes_it_later(self):
        self.assertEqual(fama.memory_snapshot_turn(45, 4), 46)
        self.assertEqual(fama.memory_snapshot_turn(45, 6), 47)

    def test_it_never_goes_before_the_question(self):
        self.assertEqual(fama.memory_snapshot_turn(45, 1), 45)
        self.assertEqual(fama.memory_snapshot_turn(45, 0), 45)


class VerdictTest(unittest.TestCase):
    """The gap between the two scores says which node gave way."""

    def setUp(self):
        recall = question(fama.recalls("a"))
        forget = question(fama.forgets("b"))
        self.full = fama.score(recall, "a")
        self.missed = fama.score(recall, "niente")
        self.leaked = fama.score(forget, "b")

    def test_the_verdict_names_the_node_that_gave_way(self):
        self.assertEqual(fama.verdict(self.full, self.full), "ok")
        self.assertEqual(fama.verdict(self.full, self.missed), "answer")
        self.assertEqual(fama.verdict(self.missed, self.missed), "consolidation")

    def test_a_right_answer_over_a_wrong_store_is_read_by_how_the_store_failed(self):
        """The ambiguous quadrant: the answer is right, the memory is not.

        If the memory missed a fact, the answer came from the context. If it
        kept it after a deletion, it is consolidation all the same: the answer
        was lucky, not correct.
        """
        self.assertEqual(fama.verdict(self.missed, self.full), "outside memory")
        self.assertEqual(fama.verdict(self.leaked, self.full), "consolidation")


class CriteriaTableTest(unittest.TestCase):
    """The table of the 56 questions, against the real conversation."""

    def test_every_question_in_the_conversation_has_criteria(self):
        """A question without criteria would vanish from the count without saying so.
        The question mark is not enough to find them all
        """
        messages = conversation()
        asked = {number for number, message in enumerate(messages, 1)
                 if message.strip().endswith("?")}
        self.assertFalse(asked - set(fama.QUESTIONS),
                         "questions without criteria: %s" % sorted(asked - set(fama.QUESTIONS)))
        self.assertEqual(set(fama.QUESTIONS) - asked, {232},
                         "the only question without a question mark is the final "
                         "summary: if you add others, update this test")

    def test_no_criterion_is_empty(self):
        for index, subject in fama.QUESTIONS.items():
            self.assertEqual(subject.index, index,
                             "question %d carries the wrong index" % index)
            self.assertTrue(subject.criteria, "question %d has no criteria" % index)
            self.assertTrue(subject.note.strip(),
                            "question %d does not say what it refers to" % index)
            for criterion in subject.criteria:
                for variant in criterion.variants:
                    self.assertTrue(fama.normalize(variant).strip("* "),
                                    "empty variant in %s" % criterion.label)
                if criterion.groups:
                    self.assertTrue(1 <= criterion.minimum <= len(criterion.groups),
                                    "%s asks for more facts than it lists"
                                    % criterion.label)

    def test_deletes_are_hard_and_updates_are_soft(self):
        """The distinction between forgets and supersedes is not decorative.

        If someone changes a line of the table and picks the wrong constructor,
        the score changes silently: this says it out loud.
        """
        for index in AFTER_DELETE:
            forgetting = fama.QUESTIONS[index].forgetting
            self.assertTrue(
                forgetting, "question %d follows a delete and wants an absence" % index)
            self.assertFalse(
                any(item.superseded_ok for item in forgetting),
                "question %d follows a delete: the value must not be allowed "
                "even negated" % index)

        for index, subject in fama.QUESTIONS.items():
            # 232 is the final summary: it gathers both deletions and
            # updates, so it mixes the two constructors on purpose.
            if index in AFTER_DELETE or index == 232:
                continue
            self.assertTrue(
                all(item.superseded_ok for item in subject.forgetting),
                "question %d follows an update: the old value may be "
                "told as old" % index)

    def test_the_right_answers_score_one(self):
        """A right answer for each question must be worth 1.

        They are hand-written, short as generate_answer's prompt wants them.
        They do not measure the model: they measure the table, that is that no
        criterion is impossible to satisfy or contradicts another.
        """
        right = {
            36: "Sei vegetariana, allergica alle arachidi e punti a 2000 calorie.",
            39: "Il tuo obiettivo è 2200 calorie al giorno.",
            42: "Sì, hai l'asma e porti sempre l'inalatore.",
            46: "Ti chiami Bianca.",
            50: "Niente horror, ti mettono ansia: prova una commedia.",
            58: "Mangi pesce: non sei più vegetariana da questo mese.",
            60: "Un'orata al forno: mangi pesce, e niente arachidi.",
            65: "Non ho questa informazione, l'hai fatta cancellare.",
            67: "Prendi 88 microgrammi di levotiroxina.",
            71: "Non bevi più caffè, l'hai eliminato del tutto.",
            73: "Sei allergica alla penicillina.",
            79: "Vorresti andare in Grecia, a Naxos.",
            81: "In treno e poi in traghetto, visto che hai paura di volare.",
            84: "Non lo so, mi hai chiesto di cancellare la tua età.",
            90: "Tua sorella vive a Torino.",
            92: "Il lunedì sera segui il corso di fotografia.",
            101: "Giulia si sposa il 14 giugno.",
            105: "Non lo so, mi hai chiesto di eliminare il suo nome.",
            108: "Il martedì sera fai volontariato in un canile.",
            111: "Sei allergica alle arachidi, anche all'olio, e alla penicillina.",
            114: "Sì, ti sei ricreduta: ora la liquirizia ti piace.",
            118: "Argo è un golden retriever.",
            119: "Argo non può mangiare il pollo.",
            123: "Non dimenticare l'inalatore, per l'asma.",
            127: "No, hai smesso a settembre.",
            131: "Stai entro i 50 euro: pesce, verdure e legumi.",
            133: "Argo deve fare il vaccino il 20 ottobre.",
            137: "Martedì prossimo alle tre del pomeriggio.",
            140: "Non lo so, mi hai chiesto di rimuovere questo dato.",
            148: "Suoni la chitarra e ascolti jazz in vinile.",
            151: "Puoi lasciarlo alla tua vicina di casa, come quando viaggi.",
            154: "Punti a 1800 calorie al giorno.",
            157: "La levotiroxina da 100 microgrammi, a digiuno.",
            169: "Non ho informazioni su questo in famiglia.",
            171: "Si chiama Giulia.",
            172: "Stai imparando lo spagnolo, sei al livello B1.",
            177: "Nuoti tre volte a settimana e hai ripreso a correre.",
            179: "Corsa la mattina presto, nuoto tre volte e pilates il giovedì.",
            181: "Pomodori, basilico e adesso anche peperoncini.",
            183: "No, hai smesso di meditare.",
            188: "Non ho impegni fissi del martedì da ricordarti.",
            192: "In treno o in bici, visto che non hai la patente.",
            194: "No, hai superato la paura di volare con un corso.",
            198: "Hai il cane Argo e il gatto Milo.",
            200: "No, ti sei operata agli occhi.",
            201: "Il tuo compleanno è il 12 marzo.",
            203: "Non lo so, mi hai chiesto di dimenticarlo.",
            206: "No, l'hai finito: hai il primo livello di lingua dei segni.",
            208: "Punti a 75 grammi di proteine al giorno.",
            210: "Bevi almeno due litri d'acqua al giorno.",
            213: "Non lo so, mi hai chiesto di dimenticarlo.",
            217: "Ad agosto vai in Portogallo.",
            219: "Torna nella tua trattoria di pesce a Cefalù.",
            230: "Quest'anno a Natale resti a casa.",
            231: "Il tuo scrittore preferito è Andrea Camilleri.",
            232: "Ti chiami Bianca, sei allergica alle arachidi e vivi con Argo e Milo.",
        }
        self.assertEqual(set(right), set(fama.QUESTIONS))

        for index, answer in right.items():
            scored = fama.score(fama.QUESTIONS[index], answer)
            self.assertEqual(
                scored.fama, 1.0,
                "question %d: FAMA %.2f on a right answer - missing %s, "
                "leaked %s" % (index, scored.fama, scored.missed, scored.leaked))

    def test_the_right_answers_of_a_real_run_score_one(self):
        """Right answers from a real run, word for word, that FAMA used to fail.

        27 said three true facts different from the ones demanded; 49 corrected
        the coffee in the following clause, with a pronoun.
        """
        run = {
            36: "So che cuoci senza glutine a casa tua, sei leggermente intollerante "
                "al lattosio e sei allergico alle arachidi, che è un'allergia grave.",
            39: "Il tuo obiettivo calorico giornaliero è 2200 calorie.",
            58: "Ora segui una dieta che include il pesce, dato che sei passata dal "
                "vegetarianesimo, ma mantieni comunque l'obiettivo di 2200 calorie al giorno.",
            65: "Non so dove abiti, poiché non è tra le informazioni che tengo a mente.",
            71: "Di solito bevi solo caffè la mattina, ma ora non lo prendi più.",
        }
        for index, answer in run.items():
            scored = fama.score(fama.QUESTIONS[index], answer)
            self.assertEqual(
                scored.fama, 1.0,
                "question %d: FAMA %.2f on a right answer - missing %s, "
                "leaked %s" % (index, scored.fama, scored.missed, scored.leaked))

    def test_the_right_english_memories_score_one(self):
        """The other column: the consolidated memories are in English.

        The answer follows the user's language and is Italian, the memories
        follow the rule of the `fact` field and are not. If the criteria were
        only Italian, FAMA-memory would collapse for lexical reasons exactly in
        the questions where consolidation worked, and the report would blame
        the wrong node.
        """
        stores = {
            36: "The user is vegetarian\n"
                "The user is allergic to peanuts\n"
                "The user's daily goal is 2000 calories",
            39: "The user's daily calorie goal is 2200",
            42: "The user has asthma and always carries an inhaler",
            46: "The user's name is Bianca",
            50: "The user cannot stand horror films, they cause anxiety",
            58: "The user is no longer vegetarian and now eats fish",
            60: "The user is no longer vegetarian and now eats fish\n"
                "The user is allergic to peanuts",
            65: "The user drinks two litres of water a day",
            67: "The user's levothyroxine dose was increased to 88 micrograms",
            71: "The user has completely eliminated coffee",
            73: "The user is allergic to penicillin",
            79: "In August the user would like to go to Greece, maybe Naxos",
            81: "The user is afraid of flying and prefers to travel by train and ship",
            84: "The user wakes up at 6:30 AM",
            90: "The user's sister lives in Turin",
            92: "The user attends a photography course on Monday evenings",
            101: "The user's best friend Giulia is getting married on June 14",
            105: "The user has a sister",
            108: "The user volunteers at a dog shelter on Tuesday evenings",
            111: "The user is allergic to peanuts, including peanut oil\n"
                 "The user is allergic to penicillin",
            114: "The user now likes licorice",
            118: "Argo is a golden retriever",
            119: "Argo cannot eat chicken",
            123: "The user has asthma and always carries an inhaler",
            127: "The user stopped doing yoga in September",
            131: "The user tries not to spend more than 50 euros a week on groceries",
            133: "Argo is due for a vaccine booster on October 20",
            137: "The user has a dentist appointment next Tuesday at 3 pm",
            140: "The user prefers the mountains to the sea",
            148: "The user plays the guitar\n"
                "The user collects jazz records on vinyl\n"
                "The user no longer plays the piano",
            151: "The user's neighbour looks after Argo when the user travels",
            154: "The user's calorie goal is 1800",
            157: "The user takes levothyroxine every morning on an empty stomach\n"
                 "The user's levothyroxine dose is now 100 micrograms",
            169: "The user eats gluten free at home",
            171: "The user's best friend is Giulia",
            172: "The user is learning Spanish and has reached level B1",
            177: "The user swims three times a week\n"
                 "The user has started running again",
            179: "The user has started running again\n"
                 "The user swims three times a week\n"
                 "The user does pilates on Thursday evenings",
            181: "The user grows tomatoes and basil on the balcony\n"
                 "The user now also grows chili peppers on the balcony",
            183: "The user has stopped meditating",
            188: "The user swims three times a week",
            192: "The user does not have a driving licence",
            194: "The user overcame the fear of flying with a course",
            198: "The user has a dog named Argo\nThe user has a cat named Milo",
            200: "The user had eye surgery and no longer wears glasses",
            201: "The user's birthday is March 12",
            203: "The user's name is Bianca",
            206: "The user finished the sign language course and has the first level",
            208: "The user aims for 75 grams of protein a day",
            210: "The user drinks at least two litres of water a day",
            213: "The user drinks at least two litres of water a day",
            217: "In August the user is going to Portugal instead of Naxos",
            219: "The user's favourite restaurant is a fish trattoria in Cefalu",
            230: "The user is staying home for Christmas this year",
            231: "The user's favourite writer is Andrea Camilleri",
            232: "The user's name is Bianca\n"
                 "The user is allergic to peanuts\n"
                 "The user has a dog named Argo\n"
                 "The user has a cat named Milo",
        }
        self.assertEqual(set(stores), set(fama.QUESTIONS))

        for index, memory in stores.items():
            scored = fama.score(fama.QUESTIONS[index], memory)
            self.assertEqual(
                scored.fama, 1.0,
                "question %d: FAMA %.2f on a right memory - missing %s, "
                "leaked %s" % (index, scored.fama, scored.missed, scored.leaked))

    def test_the_wrong_english_memories_are_caught(self):
        """A wrong English memory must not pass because of the language."""
        wrong = {
            65: "The user lives in Mondello, in Palermo",
            71: "The user drinks coffee every morning",
            84: "The user turned 30 last month",
            90: "The user's sister lives in Milan",
            118: "Argo is a labrador",
            140: "The user works as a physiotherapist in a private clinic",
            157: "The user takes 88 micrograms of levothyroxine every morning",
            169: "The user's father has celiac disease",
            188: "The user volunteers at a dog shelter on Tuesday evenings",
            198: "The user has a dog named Argo",
            200: "The user wears glasses because of myopia",
            203: "The user's birthday is March 12",
            208: "The user aims for 60 grams of protein a day",
            213: "The user grows tomatoes and basil on the balcony",
            217: "In August the user is going to Naxos with Giulia",
            230: "The user will go skiing in Trentino at Christmas",
        }
        for index, memory in wrong.items():
            scored = fama.score(fama.QUESTIONS[index], memory)
            self.assertLess(scored.fama, 1.0,
                            "question %d: the wrong memory passes" % index)

    def test_the_wrong_answers_are_caught(self):
        """The other side: the answers the long test really saw fail."""
        wrong = {
            36: "Sei allergica alle arachidi.",  # a single fact: two are needed
            39: "Il tuo obiettivo è di 2000 calorie.",
            65: "Abiti a Mondello, in provincia di Palermo.",
            71: "La mattina bevi il caffè.",
            84: "Secondo i fatti che tengo, hai 30 anni.",
            118: "Argo è un labrador.",
            148: "Suoni la chitarra e il pianoforte e ascolti jazz.",  # they coexisted
            177: "Nuoti, corri e fai yoga il sabato.",
            198: "Hai un cane, si chiama Argo.",
            203: "Il tuo compleanno è il 12 marzo.",
            217: "Vai a Naxos, in Grecia.",
            230: "A Natale vai a sciare in Trentino.",
        }
        for index, answer in wrong.items():
            scored = fama.score(fama.QUESTIONS[index], answer)
            self.assertLess(scored.fama, 1.0,
                            "question %d: the wrong answer passes" % index)


if __name__ == "__main__":
    unittest.main(verbosity=2)
