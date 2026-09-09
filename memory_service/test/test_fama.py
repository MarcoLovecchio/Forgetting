"""Il calcolo di FAMA, verificato senza toccare il modello.

Il test lungo costa una run di venti minuti e non e' ripetibile: se la metrica
che lo giudica ha un errore, l'errore si scopre in fondo e si paga di nuovo.
Qui invece la formula, il match e la tabella dei criteri si controllano in
mezzo secondo, contro l'esempio del paper e contro i casi che quel match
sbaglierebbe volentieri - il valore vecchio nominato come vecchio, e il valore
vecchio nominato dopo una negazione che riguardava un'altra cosa.

Esecuzione: python memory_service/run_tests.py -v
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

# Le domande poste subito dopo una cancellazione. Il valore non deve comparire
# nemmeno negato: la memoria non c'e' piu', nominarla e' gia' una perdita.
AFTER_DELETE = (45, 56, 68, 88, 103)


def conversation():
    """CONVERSATION letta dal sorgente, senza importare il modulo.

    Importarlo tirerebbe dentro langgraph e lo stack vero solo per leggere una
    lista di stringhe, e questo test deve girare anche dove non ci sono.
    """
    with open(LONGRUN, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "CONVERSATION":
                return ast.literal_eval(node.value)
    raise AssertionError("CONVERSATION non trovata in test_long_term_interaction.py")


def question(*criteria):
    """Una domanda di comodo, per provare la formula senza la tabella vera."""
    return fama.EvaluationQuestion(0, "prova", tuple(criteria))


class FormulaTest(unittest.TestCase):
    """FAMA = max(0, MPA - lam * (1 - FAA)), con lam = N_forget / N_totale."""

    def test_the_paper_example_weights_forgetting_by_its_share(self):
        """5 criteri di presenza e 8 di assenza: lam = 8/13, come in Memora.

        E' la domanda activity_todos_158 del persona academic_researcher,
        l'unica riportata per esteso nel repo: cinque task che devono esserci,
        otto cancellati che non devono.
        """
        subject = question(
            *[fama.recalls("task %d" % n, "task%d" % n) for n in range(5)],
            *[fama.forgets("vecchio %d" % n, "vecchio%d" % n) for n in range(8)],
        )

        perfect = fama.score(subject, "task0 task1 task2 task3 task4")
        self.assertEqual(perfect.lam, 8 / 13)
        self.assertEqual(perfect.fama, 1.0)

        # Tutta la memoria, nessuna dimenticanza: la penalita' e' proprio lam.
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
        """Le domande dopo un delete: niente da dire, solo da tacere."""
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

    def test_a_question_without_any_criterion_scores_zero(self):
        """Come il riferimento: non vale tutto per vacuita', non vale niente."""
        scored = fama.score(question(), "qualunque cosa")
        self.assertEqual(scored.lam, 0.0)
        self.assertEqual(scored.fama, 0.0)

    def test_the_declared_deviation_on_an_empty_presence_set(self):
        """Le domande dopo un delete valgono FAA, non zero fisso.

        Il codice di Memora mette MPA a 0 quando non ci sono criteri di
        presenza, e con lam = 1 quelle domande valgono 0 comunque vada. Nel loro
        dataset il ramo non gira mai; qui girerebbe sei volte su diciassette e
        bloccherebbe il totale sotto i 65 punti a prescindere. Se qualcuno
        allinea `score` alla lettera del riferimento, questo test lo ferma e gli
        dice perche'.
        """
        subject = question(fama.forgets("indirizzo", "mondello"))
        self.assertEqual(fama.score(subject, "Non lo so.").fama, 1.0)
        self.assertEqual(fama.score(subject, "Abiti a Mondello.").fama, 0.0)

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
        """Sui criteri, non sulle domande: una da sette criteri pesa sette.

        E' come li calcola `overall_metrics` del riferimento. Con la media per
        domanda gli stessi due esiti darebbero 66.7 invece di 50.0, e una
        domanda con un solo criterio sbagliato peserebbe come una con sette
        criteri giusti.
        """
        scores = [
            fama.score(question(fama.recalls("a")), "a"),
            fama.score(question(fama.recalls("a"), fama.recalls("b"),
                                fama.recalls("c")), "a"),
        ]
        overall = fama.aggregate(scores)
        self.assertEqual(overall["mpa"], 50.0)          # 2 criteri passati su 4
        self.assertIsNone(overall["faa"])               # nessun criterio di assenza
        self.assertAlmostEqual(overall["fama"], 100.0 * (1 + 1 / 3) / 2)

    def test_the_overall_accuracy_pools_both_families(self):
        """L'`overall_accuracy` del riferimento: criteri passati sul totale."""
        scores = [fama.score(question(fama.recalls("a"), fama.forgets("b")), "a b")]
        overall = fama.aggregate(scores)
        self.assertEqual(overall["mpa"], 100.0)
        self.assertEqual(overall["faa"], 0.0)
        self.assertEqual(overall["accuracy"], 50.0)     # 1 criterio su 2
        self.assertEqual(overall["fama"], 50.0)

    def test_aggregate_of_nothing_says_nothing(self):
        empty = fama.aggregate([])
        self.assertEqual(empty["questions"], 0)
        self.assertEqual(empty["criteria"], 0)
        self.assertIsNone(empty["mpa"])
        self.assertIsNone(empty["faa"])


class MatchingTest(unittest.TestCase):
    """Come si riconosce che un valore compare in un testo."""

    def test_accents_apostrophes_and_hyphens_are_the_same_word(self):
        """Le grafie del caffe' devono coincidere, o meta' criteri e' cieca."""
        self.assertEqual(fama.normalize("Caffè"), "caffe")
        self.assertEqual(fama.normalize("caffe'"), "caffe")
        self.assertEqual(fama.normalize("part-time"), "part time")

        criterion = fama.recalls("caffe", "caffe")
        self.assertTrue(criterion.satisfied("Bevi il caffè la mattina."))
        self.assertTrue(criterion.satisfied("bevi il caffe' la mattina"))

    def test_a_trailing_star_matches_the_whole_family(self):
        criterion = fama.recalls("corsa", "corr*")
        self.assertTrue(criterion.satisfied("Corri tre volte a settimana."))
        self.assertTrue(criterion.satisfied("Hai ripreso a correre."))
        self.assertFalse(criterion.satisfied("Fai nuoto."))

    def test_a_star_in_the_middle_bridges_the_words_between(self):
        """'non*piu' deve prendere 'non fai piu'', che e' lo stesso modo di dire."""
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
    """Cancellare e superare non sono la stessa cosa, e i criteri lo dicono."""

    def test_after_a_delete_the_value_must_not_appear_even_negated(self):
        """Dire 'non abiti piu' a Mondello' rivela quello che doveva sparire."""
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
        """La negazione vale nella sua proposizione, non in tutta la risposta.

        Senza questo, "Non bevi alcolici. Bevi caffe' la mattina." passerebbe:
        la parola "non" c'e', ma nega un'altra cosa.
        """
        criterion = fama.supersedes("caffe'", "caffe")
        self.assertFalse(criterion.satisfied("Non bevi alcolici. Bevi caffè la mattina."))
        self.assertFalse(criterion.satisfied("Non fai più yoga, e bevi caffè la mattina."))
        self.assertTrue(criterion.satisfied("Non bevi più caffè."))

    def test_english_negations_supersede_too(self):
        """Le memorie sono inglesi: senza questi marcatori sarebbero tutte fughe."""
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
        """L'unico marcatore inglese che collide con l'italiano e' "no".

        In inglese apre "no longer"; in italiano e' la risposta secca, e senza
        il taglio sulla punteggiatura coprirebbe tutta la frase che segue -
        "No, tua sorella vive a Milano" passerebbe per un superamento.

        Resta un buco noto: "No tua sorella vive a Milano" senza virgola e'
        un'unica proposizione e passa. Serve una risposta sgrammaticata per
        incontrarlo, e chiudere la falla vorrebbe dire togliere "no", che serve
        a tutte le memorie inglesi.
        """
        criterion = fama.supersedes("vecchia citta'", "milano", "milan")
        self.assertFalse(criterion.satisfied("No, tua sorella vive a Milano."))
        self.assertFalse(criterion.satisfied("No. Tua sorella vive a Milano."))
        self.assertTrue(criterion.satisfied("No, si è trasferita a Torino da Milano."))

    def test_the_present_tense_is_not_a_marker_in_either_language(self):
        """"ora" e "now" stanno nella frase del valore NUOVO, non del vecchio."""
        criterion = fama.supersedes("indirizzo", "mondello")
        self.assertFalse(criterion.satisfied("The user now lives in Mondello."))
        self.assertFalse(criterion.satisfied("Adesso abiti a Mondello."))

    def test_all_occurrences_must_be_acceptable_not_just_one(self):
        criterion = fama.supersedes("caffe'", "caffe")
        self.assertFalse(criterion.satisfied(
            "Non bevi più caffè. Il caffè lo prendi alle otto."))


class VerdictTest(unittest.TestCase):
    """Lo scarto fra i due punteggi dice quale nodo ha ceduto."""

    def setUp(self):
        recall = question(fama.recalls("a"))
        forget = question(fama.forgets("b"))
        self.full = fama.score(recall, "a")
        self.missed = fama.score(recall, "niente")
        self.leaked = fama.score(forget, "b")

    def test_the_verdict_names_the_node_that_gave_way(self):
        self.assertEqual(fama.verdict(self.full, self.full), "ok")
        self.assertEqual(fama.verdict(self.full, self.missed), "risposta")
        self.assertEqual(fama.verdict(self.missed, self.missed), "consolidation")

    def test_a_right_answer_over_a_wrong_store_is_read_by_how_the_store_failed(self):
        """Il quadrante ambiguo: la risposta e' giusta, la memoria no.

        Se la memoria ha mancato un fatto, la risposta veniva dal contesto. Se
        se l'e' tenuto dopo una cancellazione, e' consolidation lo stesso: la
        risposta e' stata fortunata, non corretta.
        """
        self.assertEqual(fama.verdict(self.missed, self.full), "fuori memoria")
        self.assertEqual(fama.verdict(self.leaked, self.full), "consolidation")


class CriteriaTableTest(unittest.TestCase):
    """La tabella delle diciassette domande, contro la conversazione vera."""

    def test_every_annotated_index_is_inside_the_conversation(self):
        messages = conversation()
        for index in fama.QUESTIONS:
            self.assertTrue(1 <= index <= len(messages),
                            "indice %d fuori dalla conversazione" % index)

    def test_every_question_in_the_conversation_has_criteria(self):
        """Una domanda senza criteri sparirebbe dal conto senza dirlo.

        Il punto interrogativo non basta a trovarle tutte: il msg 117
        ("Riassumi tutto quello che sai di me.") chiede alla memoria di
        parlare esattamente come le altre, e non ne ha uno. E' la ragione per
        cui la tabella, e non la punteggiatura, decide cosa viene valutato.
        """
        messages = conversation()
        asked = {number for number, message in enumerate(messages, 1)
                 if message.strip().endswith("?")}
        self.assertFalse(asked - set(fama.QUESTIONS),
                         "domande senza criteri: %s" % sorted(asked - set(fama.QUESTIONS)))
        self.assertEqual(set(fama.QUESTIONS) - asked, {117},
                         "l'unica domanda senza punto interrogativo e' il riassunto "
                         "finale: se ne aggiungi altre, aggiorna questo test")

    def test_no_criterion_is_empty(self):
        for index, subject in fama.QUESTIONS.items():
            self.assertEqual(subject.index, index,
                             "la domanda %d porta l'indice sbagliato" % index)
            self.assertTrue(subject.criteria, "la domanda %d non ha criteri" % index)
            self.assertTrue(subject.note.strip(),
                            "la domanda %d non dice a cosa si riferisce" % index)
            for criterion in subject.criteria:
                self.assertTrue(criterion.variants,
                                "criterio senza varianti: %s" % criterion.label)
                for variant in criterion.variants:
                    self.assertTrue(fama.normalize(variant).strip("* "),
                                    "variante vuota in %s" % criterion.label)

    def test_deletes_are_hard_and_updates_are_soft(self):
        """La distinzione fra forgets e supersedes non e' decorativa.

        Se qualcuno cambia una riga della tabella e sbaglia costruttore, il
        punteggio cambia in silenzio: questo lo dice ad alta voce.
        """
        for index in AFTER_DELETE:
            forgetting = fama.QUESTIONS[index].forgetting
            self.assertTrue(
                forgetting, "la domanda %d segue un delete e vuole un'assenza" % index)
            self.assertFalse(
                any(item.superseded_ok for item in forgetting),
                "la domanda %d segue un delete: il valore non va ammesso "
                "nemmeno negato" % index)

        for index, subject in fama.QUESTIONS.items():
            # 117 e' il riassunto finale: raccoglie sia cancellazioni sia
            # aggiornamenti, quindi mescola i due costruttori apposta.
            if index in AFTER_DELETE or index == 117:
                continue
            self.assertTrue(
                all(item.superseded_ok for item in subject.forgetting),
                "la domanda %d segue un update: il valore vecchio puo' essere "
                "raccontato come vecchio" % index)

    def test_the_right_answers_score_one(self):
        """Una risposta giusta per ogni domanda deve valere 1.

        Sono scritte a mano, corte come le vuole il prompt di generate_answer.
        Non misurano il modello: misurano la tabella, cioe' che nessun criterio
        sia impossibile da soddisfare o si contraddica con un altro.
        """
        right = {
            27: "Sei vegetariana, allergica alle arachidi e punti a 2000 calorie.",
            30: "Il tuo obiettivo è 2200 calorie al giorno.",
            40: "Mangi pesce: non sei più vegetariana da questo mese.",
            45: "Non ho questa informazione, l'hai fatta cancellare.",
            49: "Non bevi più caffè, l'hai eliminato del tutto.",
            56: "Non lo so, mi hai chiesto di cancellare la tua età.",
            60: "Tua sorella vive a Torino.",
            68: "Non lo so, mi hai chiesto di eliminare il suo nome.",
            76: "Argo è un golden retriever.",
            81: "No, hai smesso a settembre.",
            88: "Non lo so, mi hai chiesto di rimuovere questo dato.",
            92: "Suoni la chitarra e ascolti jazz in vinile.",
            95: "Punti a 1800 calorie al giorno.",
            103: "Non ho informazioni su questo in famiglia.",
            107: "Nuoti tre volte a settimana e hai ripreso a correre.",
            113: "Hai il cane Argo e il gatto Milo.",
            117: "Ti chiami Bianca, sei allergica alle arachidi e vivi con Argo e Milo.",
        }
        self.assertEqual(set(right), set(fama.QUESTIONS))

        for index, answer in right.items():
            scored = fama.score(fama.QUESTIONS[index], answer)
            self.assertEqual(
                scored.fama, 1.0,
                "domanda %d: FAMA %.2f su una risposta giusta - mancanti %s, "
                "trapelati %s" % (index, scored.fama, scored.missed, scored.leaked))

    def test_the_right_english_memories_score_one(self):
        """L'altra colonna: le memorie consolidate sono in inglese.

        La risposta segue la lingua dell'utente ed e' italiana, le memorie
        seguono la regola del campo `fact` e non lo sono. Se i criteri fossero
        solo italiani, FAMA-memoria crollerebbe per motivi lessicali proprio
        nelle domande dove la consolidation ha funzionato, e il resoconto
        accuserebbe il nodo sbagliato.
        """
        stores = {
            27: "The user is vegetarian\n"
                "The user is allergic to peanuts\n"
                "The user's daily goal is 2000 calories",
            30: "The user's daily calorie goal is 2200",
            40: "The user is no longer vegetarian and now eats fish",
            45: "The user drinks two litres of water a day",
            49: "The user has completely eliminated coffee",
            56: "The user has a dog named Argo",
            60: "The user's sister lives in Turin",
            68: "The user has a sister",
            76: "Argo is a golden retriever",
            81: "The user stopped doing yoga in September",
            88: "The user prefers the mountains to the sea",
            92: "The user plays the guitar\n"
                "The user collects jazz records on vinyl\n"
                "The user no longer plays the piano",
            95: "The user's calorie goal is 1800",
            103: "The user eats gluten free at home",
            107: "The user swims three times a week\n"
                 "The user has started running again",
            113: "The user has a dog named Argo\nThe user has a cat named Milo",
            117: "The user's name is Bianca\n"
                 "The user is allergic to peanuts\n"
                 "The user has a dog named Argo\n"
                 "The user has a cat named Milo",
        }
        self.assertEqual(set(stores), set(fama.QUESTIONS))

        for index, memory in stores.items():
            scored = fama.score(fama.QUESTIONS[index], memory)
            self.assertEqual(
                scored.fama, 1.0,
                "domanda %d: FAMA %.2f su una memoria giusta - mancanti %s, "
                "trapelati %s" % (index, scored.fama, scored.missed, scored.leaked))

    def test_the_wrong_english_memories_are_caught(self):
        """Una memoria inglese sbagliata non deve passare per la lingua."""
        wrong = {
            45: "The user lives in Mondello, in Palermo",
            49: "The user drinks coffee every morning",
            60: "The user's sister lives in Milan",
            76: "Argo is a labrador",
            88: "The user works as a physiotherapist in a private clinic",
            103: "The user's father has celiac disease",
            113: "The user has a dog named Argo",
        }
        for index, memory in wrong.items():
            scored = fama.score(fama.QUESTIONS[index], memory)
            self.assertLess(scored.fama, 1.0,
                            "domanda %d: la memoria sbagliata passa" % index)

    def test_the_wrong_answers_are_caught(self):
        """L'altro lato: le risposte che il test lungo ha visto davvero sbagliare."""
        wrong = {
            30: "Il tuo obiettivo è di 2000 calorie.",
            45: "Abiti a Mondello, in provincia di Palermo.",
            49: "La mattina bevi il caffè.",
            76: "Argo è un labrador.",
            113: "Hai un cane, si chiama Argo.",
        }
        for index, answer in wrong.items():
            scored = fama.score(fama.QUESTIONS[index], answer)
            self.assertLess(scored.fama, 1.0,
                            "domanda %d: la risposta sbagliata passa" % index)


if __name__ == "__main__":
    unittest.main(verbosity=2)
