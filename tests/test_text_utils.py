"""Normalisation rules (one test per rule), tokenizer, sentence splitter and truncation."""

from src.text_utils import (
    count_words,
    normalize_text,
    split_at_word_limit,
    split_sentences,
    tokenize,
    truncate_words,
)


def test_rule1_control_whitespace_becomes_space():
    assert normalize_text("one\ntwo\r\nthree\tfour") == "one two three four"


def test_rule2_no_space_before_closing_punctuation():
    assert normalize_text("word . next , more ! why ? semi ; colon : 50 % (a ) [b ] {c }") == \
        "word. next, more! why? semi; colon: 50% (a) [b] {c}"


def test_rule3_no_space_after_opening_brackets():
    assert normalize_text("( a) [ b] { c}") == "(a) [b] {c}"


def test_rule4_contractions_reattached_case_insensitive():
    text = "I do n't know , it 's fine , they 're here , we 've gone , I 'll go , he 'd say , I 'm OK , DO N'T"
    assert normalize_text(text) == (
        "I don't know, it's fine, they're here, we've gone, I'll go, he'd say, I'm OK, DON'T"
    )


def test_rule4_typographic_apostrophe_and_quotes_untouched():
    assert normalize_text("do n’t") == "don't"  # rule 9 maps U+2019 first, then rule 4 re-attaches
    # A quoted word starting with 's' is not a contraction.
    assert normalize_text("he said 'so what'") == "he said 'so what'"


BS = "\\"


def test_rule1b_escaped_line_breaks():
    assert normalize_text("infections." + BS + "n" + BS + "nHelps regulate") == "infections. Helps regulate"
    assert normalize_text("end." + BS + "r" + BS + "n" + BS + "tNext") == "end. Next"
    assert normalize_text("questions." + BS + "n" + BS + "n") == "questions."
    # LaTeX commands that start with n, r or t are kept.
    latex = "{" + BS + "displaystyle " + BS + "neg " + BS + "theta x}"
    assert normalize_text(latex) == latex


def test_rule9_curly_apostrophe():
    assert normalize_text("I don’t know, it’s Sam’s") == "I don't know, it's Sam's"


def test_rule10_spaced_slash():
    assert normalize_text("60 km / h and /u / name") == "60 km/h and /u/name"
    assert normalize_text("a / b") == "a/b"
    assert normalize_text("ratio 1 /") == "ratio 1 /"  # no word character after it


def test_rule11_detached_single_quotes():
    text = "When we say ' prove ' we mean ' show that something is true '."
    assert normalize_text(text) == "When we say 'prove' we mean 'show that something is true'."
    assert normalize_text("the judges ' vision") == "the judges' vision"  # odd count: plural possessive
    assert normalize_text("makin ' bank") == "makin' bank"  # odd count: elision
    assert normalize_text("Phelps ' death and ' quoted words ' here") == "Phelps' death and 'quoted words' here"
    # Odd count without a possessive candidate is left alone; normal quotes are untouched.
    assert normalize_text("In addition to ' kcell's response.") == "In addition to ' kcell's response."
    assert normalize_text("it's 'fine' and Sam's") == "it's 'fine' and Sam's"


def test_rule12_glued_sentence_boundaries():
    assert normalize_text("in certain situations.One potential") == "in certain situations. One potential"
    assert normalize_text("Hello!My name is Dr. Aida") == "Hello! My name is Dr. Aida"
    assert normalize_text("Really?Yes.") == "Really? Yes."
    assert normalize_text("the (CPI).The rate") == "the (CPI). The rate"
    assert normalize_text("Form 1040.It is") == "Form 1040. It is"
    # Abbreviations, initials and decimals are not split.
    assert normalize_text("the U.S.A and J.K.Rowling, 3.5 GHz") == "the U.S.A and J.K.Rowling, 3.5 GHz"
    assert split_sentences(normalize_text("situations.One more.Two")) == ["situations.", "One more.", "Two"]


def test_extra_rules_are_idempotent():
    text = ("say ' prove ' km / h situations.One don’t" + BS + "n" + BS + "nNext judges ' vision")
    once = normalize_text(text)
    assert normalize_text(once) == once


def test_rule5_spaced_hyphen_between_words():
    assert normalize_text("an oscar - winning film") == "an oscar-winning film"
    assert normalize_text("ages 5 - 7") == "ages 5-7"


def test_rule6_spaced_double_quotes():
    assert normalize_text('the " Best Seller " list') == 'the "Best Seller" list'


def test_rule6_quotes_paired_left_to_right():
    assert normalize_text('a "b" c "d"') == 'a "b" c "d"'


def test_rule7_hash_number():
    assert normalize_text("ranked # 1 and # 23") == "ranked #1 and #23"


def test_rule8_collapse_and_strip():
    assert normalize_text("   many    spaces   here  ") == "many spaces here"


def test_normalize_handles_empty_and_is_idempotent():
    assert normalize_text("") == ""
    assert normalize_text(None) == ""
    text = "I do n't know . It 's ( maybe ) a \" test \" - like # 1 thing .\n\nNext"
    once = normalize_text(text)
    assert normalize_text(once) == once


def test_hc3_artefacts_removed():
    text = "you ca n't see , and it 's dark . So"
    out = normalize_text(text)
    assert " ." not in out and " ," not in out and " n't" not in out and "  " not in out


def test_tokenizer_fixed_pattern():
    assert tokenize("Don't stop: 3.5 apples, 42 pears; e-mail") == ["don't", "stop", "3.5", "apples", "42",
                                                                    "pears", "e", "mail"]
    # The tokenizer is exactly the specified pattern; the typographic apostrophe is handled by normalisation.
    assert tokenize("don’t") == ["don", "t"]
    assert tokenize(normalize_text("don’t")) == ["don't"]
    assert count_words("Hello, world!") == 2


def test_sentence_splitter():
    assert split_sentences("One two. Three four! Five? Six") == ["One two.", "Three four!", "Five?", "Six"]
    assert split_sentences("") == []
    assert split_sentences("Decimal 3.5 stays.") == ["Decimal 3.5 stays."]


def test_truncate_words_keeps_punctuation_and_limit():
    text = "Alpha, beta. Gamma delta! Epsilon zeta."
    assert truncate_words(text, 3) == "Alpha, beta. Gamma"
    assert truncate_words(text, 100) == text
    assert truncate_words(text, 0) == ""


def test_truncate_words_never_exceeds_limit():
    text = "one two well-known three"
    out = truncate_words(text, 3)
    assert count_words(out) <= 3
    assert out == "one two"


def test_split_at_word_limit():
    head, rest = split_at_word_limit("a b c d e", 3)
    assert head == "a b c"
    assert rest == "d e"
