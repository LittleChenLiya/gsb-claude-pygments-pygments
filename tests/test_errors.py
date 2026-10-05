"""
    Lexer error tracking tests
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    Tests for the explicit error list provided by `LexerErrors` and the
    `errors` parameter of `Lexer.get_tokens()`.

    :copyright: Copyright 2006-present by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

import pytest

from pygments.lexer import (
    ErrorRecord,
    ExtendedRegexLexer,
    Lexer,
    LexerErrors,
    RegexLexer,
    combined,
    lex_incremental,
)
from pygments.token import Error, String, Text


class MyLexer(RegexLexer):
    """A simple lexer with a string state and a combined state."""
    tokens = {
        'root': [
            ('a', Text.Root),
            ('b', Text.Root),
            ('"', String, 'string'),
            ('c', Text.Root, combined('s1', 's2')),
        ],
        'string': [
            ('"', String, '#pop'),
            ('x', String),
        ],
        's1': [
            ('1', Text.S1),
        ],
        's2': [
            ('2', Text.S2),
        ],
    }


class BoomLexer(RegexLexer):
    """A lexer that raises an exception while lexing."""
    tokens = {
        'root': [
            ('a', Text),
        ],
    }

    def get_tokens_unprocessed(self, text, stack=('root',)):
        raise ValueError('boom')
        yield  # pragma: no cover


class PlainLexer(Lexer):
    """A non-RegexLexer lexer that emits Error tokens directly."""

    def get_tokens_unprocessed(self, text):
        for i, ch in enumerate(text):
            if ch == 'x':
                yield i, Error, ch
            else:
                yield i, Text, ch


@pytest.fixture(scope='module')
def lexer():
    return MyLexer()


def test_basic_error_record(lexer):
    code = 'axb\n'
    errors = LexerErrors()
    tokens = list(lexer.get_tokens(code, errors=errors))
    assert any(t is Error for t, _ in tokens)
    assert errors.records == [
        ErrorRecord(line=1, column=2, state='root', pos=1, end=2)]
    assert errors.exception is None


def test_error_tokens_not_deleted(lexer):
    # Normal Error tokens must still be yielded.
    code = 'xx\n'
    errors = LexerErrors()
    tokens = list(lexer.get_tokens(code, errors=errors))
    error_tokens = [v for t, v in tokens if t is Error]
    assert error_tokens == ['x', 'x']
    assert len(errors.records) == 2


def test_no_text_stored(lexer):
    # The callback only sees type and position; the text is not stored.
    code = 'axb\n'
    errors = LexerErrors()
    list(lexer.get_tokens(code, errors=errors))
    record = errors.records[0]
    assert not hasattr(record, 'text')
    assert record.pos == 1
    assert record.end == 2


def test_line_col_multiline(lexer):
    code = 'a\nbx\n'
    errors = LexerErrors()
    list(lexer.get_tokens(code, errors=errors))
    assert errors.records == [
        ErrorRecord(line=2, column=2, state='root', pos=3, end=4)]


def test_state_recorded(lexer):
    # Errors in the string state record the state name.
    code = 'a"bad"\n'
    errors = LexerErrors()
    list(lexer.get_tokens(code, errors=errors))
    states = [r.state for r in errors.records]
    assert states == ['string', 'string', 'string']


def test_combined_state_not_public(lexer):
    # Combined states use generated private names; they are recorded as None.
    code = 'c3\n'
    errors = LexerErrors()
    list(lexer.get_tokens(code, errors=errors))
    assert all(r.state is None for r in errors.records)


def test_merge_adjacent(lexer):
    code = 'xxaxx\n'
    errors = LexerErrors(merge_adjacent=True)
    list(lexer.get_tokens(code, errors=errors))
    assert errors.records == [
        ErrorRecord(line=1, column=1, state='root', pos=0, end=2),
        ErrorRecord(line=1, column=4, state='root', pos=3, end=5)]


def test_no_merge_by_default(lexer):
    code = 'xx\n'
    errors = LexerErrors()
    list(lexer.get_tokens(code, errors=errors))
    assert len(errors.records) == 2
    assert errors.records[0].end == 1
    assert errors.records[1].pos == 1


def test_exception_recorded_separately():
    lexer = BoomLexer()
    errors = LexerErrors()
    with pytest.raises(ValueError, match='boom'):
        list(lexer.get_tokens('aaa', errors=errors))
    assert isinstance(errors.exception, ValueError)
    assert errors.exception_stage == 'lexing'
    # No Error token records were produced.
    assert errors.records == []


def test_exception_and_error_tokens_attributed_separately():
    # A lexer that emits Error tokens and then raises.
    class MixedLexer(RegexLexer):
        tokens = {'root': [('a', Text)]}

        def get_tokens_unprocessed(self, text, stack=('root',)):
            for i, ch in enumerate(text):
                if ch == 'x':
                    yield i, Error, ch
            raise ValueError('mixed')

    lexer = MixedLexer()
    errors = LexerErrors()
    with pytest.raises(ValueError, match='mixed'):
        list(lexer.get_tokens('axa', errors=errors))
    assert len(errors.records) == 1
    assert errors.records[0].pos == 1
    assert isinstance(errors.exception, ValueError)
    assert errors.exception_stage == 'lexing'


def test_preprocessing_exception():
    lexer = BoomLexer()
    errors = LexerErrors()
    # Force a preprocessing error by passing bytes with an invalid encoding.
    lexer.encoding = 'invalid-encoding'
    with pytest.raises(LookupError):
        list(lexer.get_tokens(b'aaa', errors=errors))
    assert errors.exception_stage == 'preprocessing'


def test_plain_lexer_errors():
    # Non-RegexLexer lexers also produce error records (with state=None).
    lexer = PlainLexer()
    errors = LexerErrors()
    tokens = list(lexer.get_tokens('axb\n', errors=errors))
    assert any(t is Error for t, _ in tokens)
    assert errors.records == [
        ErrorRecord(line=1, column=2, state=None, pos=1, end=2)]


def test_get_errors_convenience(lexer):
    errors = lexer.get_errors('axb\n')
    assert isinstance(errors, LexerErrors)
    assert len(errors.records) == 1
    assert bool(errors)
    assert list(errors) == errors.records


def test_clear(lexer):
    errors = LexerErrors()
    list(lexer.get_tokens('axb\n', errors=errors))
    assert len(errors.records) == 1
    errors.clear()
    assert errors.records == []
    assert errors.exception is None
    assert not errors
    assert len(errors) == 0


def test_line_col_method(lexer):
    errors = LexerErrors()
    errors.bind('a\nbx\n')
    assert errors.line_col(0) == (1, 1)
    assert errors.line_col(1) == (1, 2)
    assert errors.line_col(2) == (2, 1)
    assert errors.line_col(3) == (2, 2)


def test_record_asdict(lexer):
    errors = lexer.get_errors('axb\n')
    record = errors.records[0]
    assert record.asdict() == {
        'line': 1, 'column': 2, 'state': 'root', 'pos': 1, 'end': 2}


def test_record_equality(lexer):
    errors = lexer.get_errors('axb\n')
    record = errors.records[0]
    assert record == ErrorRecord(1, 2, 'root', 1, 2)
    assert record != ErrorRecord(1, 2, 'root', 1, 3)
    assert record != 'not a record'


def test_incremental_matches_one_time(lexer):
    code = 'axb\nxx\n'
    # One-time lex.
    errors1 = LexerErrors()
    list(lexer.get_tokens(code, errors=errors1))
    # Incremental lex: split at a newline boundary.
    full = lexer._preprocess_lexer_input(code)
    mid = full.index('\n') + 1
    segments = [full[:mid], full[mid:]]
    errors2 = LexerErrors()
    tokens2, _ = lex_incremental(lexer, segments, errors=errors2)
    # Error positions match.
    assert errors2.records == errors1.records
    # Token order is unchanged.
    expected = list(lexer.get_tokens(code))
    assert tokens2 == expected


def test_incremental_three_segments(lexer):
    code = 'axb\nxx\ny\n'
    errors1 = LexerErrors()
    list(lexer.get_tokens(code, errors=errors1))
    full = lexer._preprocess_lexer_input(code)
    # Split into three segments at newline boundaries.
    lines = full.splitlines(keepends=True)
    errors2 = LexerErrors()
    lex_incremental(lexer, lines, errors=errors2)
    assert errors2.records == errors1.records


def test_incremental_without_errors_arg(lexer):
    code = 'axb\n'
    full = lexer._preprocess_lexer_input(code)
    mid = full.index('\n') + 1
    segments = [full[:mid], full[mid:]]
    _, errors = lex_incremental(lexer, segments)
    assert isinstance(errors, LexerErrors)
    assert len(errors.records) == 1


def test_incremental_preserves_state(lexer):
    # Split in the middle of a string state; the state must persist.
    code = 'a"bad"a\n'
    errors1 = LexerErrors()
    list(lexer.get_tokens(code, errors=errors1))
    full = lexer._preprocess_lexer_input(code)
    # Split inside the string (after the opening quote).
    mid = full.index('"') + 1
    segments = [full[:mid], full[mid:]]
    errors2 = LexerErrors()
    lex_incremental(lexer, segments, errors=errors2)
    assert errors2.records == errors1.records


def test_extended_regex_lexer():
    class ExtLexer(ExtendedRegexLexer):
        tokens = {
            'root': [
                ('a', Text),
                ('b', Text),
            ],
        }

    lexer = ExtLexer()
    code = 'axb\n'
    errors = LexerErrors()
    list(lexer.get_tokens(code, errors=errors))
    assert errors.records == [
        ErrorRecord(line=1, column=2, state='root', pos=1, end=2)]


def test_no_errors_arg_still_works(lexer):
    # Without errors, get_tokens behaves exactly as before.
    tokens = list(lexer.get_tokens('axb\n'))
    assert any(t is Error for t, _ in tokens)


def test_lexer_errors_repr(lexer):
    errors = lexer.get_errors('axb\n')
    assert 'LexerErrors' in repr(errors)
    assert 'records' in repr(errors)
