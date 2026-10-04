"""
Tests for the chunked (incremental) lexing interface of RegexLexer:
``lex_chunk``, ``start_chunked`` and ``LexerCheckpoint``.
"""

import pickle

import pytest

from pygments.lexer import (
    CheckpointError,
    CheckpointMismatchError,
    CheckpointUnsupportedError,
    LexerCheckpoint,
    RegexLexer,
)
from pygments.lexers import PythonLexer, RustLexer
from pygments.token import Name, Text


def merge_adjacent(tokens):
    """Merge adjacent tokens of the same type into single tokens.

    A token spanning a chunk boundary (e.g. a multi-line string) is
    emitted as one token per chunk; after merging, the chunked token
    stream must equal the one-shot stream exactly.
    """
    merged = []
    for pos, tokentype, value in tokens:
        if (merged and merged[-1][1] is tokentype
                and merged[-1][0] + len(merged[-1][2]) == pos):
            prev_pos, _, prev_value = merged[-1]
            merged[-1] = (prev_pos, tokentype, prev_value + value)
        else:
            merged.append((pos, tokentype, value))
    return merged


def lex_one_shot(lexer, text):
    return merge_adjacent(list(lexer.get_tokens_unprocessed(text)))


def lex_chunked(lexer, chunks):
    checkpoint = None
    tokens = []
    for i, chunk in enumerate(chunks):
        final = i == len(chunks) - 1 and not chunk.endswith('\n')
        new_tokens, checkpoint = lexer.lex_chunk(chunk, checkpoint, final=final)
        tokens.extend(new_tokens)
    return merge_adjacent(tokens), checkpoint


# Three lines; the triple-quoted string spans the first two line
# boundaries, so the lexer is inside a nested string state at both cuts.
PYTHON_SOURCE = 'x = """hello\nworld"""\nprint(x)\n'
PYTHON_LINES = PYTHON_SOURCE.splitlines(keepends=True)
assert len(PYTHON_LINES) == 3

# Three lines with a multi-line (block) comment spanning all of them.
# RustLexer keeps the comment in a dedicated lexer state, so the state
# stack at each line boundary is non-trivial.
COMMENT_SOURCE = 'let a = 1; /* start\nmiddle\nend */ let b = 2;\n'
COMMENT_LINES = COMMENT_SOURCE.splitlines(keepends=True)
assert len(COMMENT_LINES) == 3


def test_chunked_equals_one_shot_multiline_string():
    """Feeding three lines (with a multi-line string) one by one yields
    the same token types and absolute offsets as one-shot highlighting."""
    chunked, _ = lex_chunked(PythonLexer(), PYTHON_LINES)
    assert chunked == lex_one_shot(PythonLexer(), PYTHON_SOURCE)


def test_absolute_offsets_are_not_reset():
    """Offsets of later chunks continue where the previous chunk ended."""
    lexer = PythonLexer()
    checkpoint = None
    offset = 0
    for line in PYTHON_LINES:
        tokens, checkpoint = lexer.lex_chunk(line, checkpoint)
        assert tokens[0][0] == offset
        offset += len(line)
    # and they match the one-shot offsets
    one_shot = list(PythonLexer().get_tokens_unprocessed(PYTHON_SOURCE))
    assert merge_adjacent([t for t in one_shot]) == lex_one_shot(
        PythonLexer(), PYTHON_SOURCE)


def test_checkpoint_resume_with_fresh_lexer():
    """A checkpoint saved mid-document can be resumed by a newly created
    lexer instance (after a pickle round-trip) with identical results."""
    lexer_a = PythonLexer()
    tokens, checkpoint = lexer_a.lex_chunk(PYTHON_LINES[0])

    # "save" the checkpoint: it must survive serialization and must not
    # drag the source text along
    checkpoint = pickle.loads(pickle.dumps(checkpoint))

    lexer_b = PythonLexer()
    for line in PYTHON_LINES[1:]:
        new_tokens, checkpoint = lexer_b.lex_chunk(line, checkpoint)
        tokens.extend(new_tokens)

    assert merge_adjacent(tokens) == lex_one_shot(PythonLexer(), PYTHON_SOURCE)


def test_multiline_comment_across_chunks():
    """A block comment spanning three lines is lexed consistently."""
    chunked, _ = lex_chunked(RustLexer(), COMMENT_LINES)
    assert chunked == lex_one_shot(RustLexer(), COMMENT_SOURCE)


def test_nested_state_stack_in_checkpoint():
    """Inside the block comment, the checkpoint carries the 'comment'
    state on the stack."""
    lexer = RustLexer()
    _, checkpoint = lexer.lex_chunk(COMMENT_LINES[0])
    assert 'comment' in checkpoint.stack
    _, checkpoint = lexer.lex_chunk(COMMENT_LINES[1], checkpoint)
    assert 'comment' in checkpoint.stack
    _, checkpoint = lexer.lex_chunk(COMMENT_LINES[2], checkpoint)
    assert 'comment' not in checkpoint.stack


def test_final_chunk_without_newline():
    """The last chunk may lack a trailing newline if final=True."""
    source = PYTHON_SOURCE.rstrip('\n')
    chunks = source.splitlines(keepends=True)
    chunked, checkpoint = lex_chunked(PythonLexer(), chunks)
    assert checkpoint is None  # final chunk returns no checkpoint
    assert chunked == lex_one_shot(PythonLexer(), source)


def test_chunk_must_end_at_line_boundary():
    lexer = PythonLexer()
    with pytest.raises(CheckpointError, match='line boundaries'):
        lexer.lex_chunk('x = 1')  # no trailing newline, not final
    # ... but it is fine with final=True
    tokens, checkpoint = lexer.lex_chunk('x = 1', final=True)
    assert checkpoint is None
    assert tokens == lex_one_shot(PythonLexer(), 'x = 1')


def test_checkpoint_mismatch_other_lexer_class():
    _, checkpoint = PythonLexer().lex_chunk(PYTHON_LINES[0])
    with pytest.raises(CheckpointMismatchError):
        RustLexer().lex_chunk(COMMENT_LINES[1], checkpoint)


def test_checkpoint_mismatch_other_options():
    _, checkpoint = PythonLexer().lex_chunk(PYTHON_LINES[0])
    with pytest.raises(CheckpointMismatchError):
        PythonLexer(tabsize=8).lex_chunk(PYTHON_LINES[1], checkpoint)


def test_checkpoint_mismatch_format_version():
    _, checkpoint = PythonLexer().lex_chunk(PYTHON_LINES[0])
    broken = LexerCheckpoint(checkpoint.lexer_key, checkpoint.pos,
                             checkpoint.stack, version=-1)
    with pytest.raises(CheckpointMismatchError):
        PythonLexer().lex_chunk(PYTHON_LINES[1], broken)


def test_invalid_checkpoint_object():
    with pytest.raises(CheckpointError, match='LexerCheckpoint'):
        PythonLexer().lex_chunk(PYTHON_LINES[0], checkpoint=object())


def test_unsupported_custom_get_tokens_unprocessed():
    """A lexer overriding get_tokens_unprocessed cannot checkpoint its
    custom state and must refuse explicitly."""

    class CustomLexer(RegexLexer):
        tokens = {'root': [(r'\w+', Name), (r'\s+', Text)]}

        def get_tokens_unprocessed(self, text, stack=('root',)):
            yield from super().get_tokens_unprocessed(text, stack)

    lexer = CustomLexer()
    with pytest.raises(CheckpointUnsupportedError):
        lexer.start_chunked()
    with pytest.raises(CheckpointUnsupportedError):
        lexer.lex_chunk('foo bar\n')


def test_unsupported_declared_by_lexer():
    """A lexer can declare itself unsafe for checkpointing."""

    class StatefulLexer(RegexLexer):
        supports_checkpoints = False
        tokens = {'root': [(r'\w+', Name), (r'\s+', Text)]}

    with pytest.raises(CheckpointUnsupportedError):
        StatefulLexer().lex_chunk('foo\n')


def test_checkpoint_does_not_store_source_text():
    """The checkpoint must stay small and not contain the source."""
    _, checkpoint = PythonLexer().lex_chunk(PYTHON_LINES[0])
    assert not hasattr(checkpoint, 'text')
    assert 'hello' not in repr(checkpoint)
    assert 'hello' not in repr(pickle.dumps(checkpoint))
    # only the absolute offset and the state stack are kept
    assert checkpoint.pos == len(PYTHON_LINES[0])
    assert checkpoint.stack and all(isinstance(s, str) for s in checkpoint.stack)


def test_start_chunked_matches_implicit_start():
    lexer = PythonLexer()
    explicit = lexer.start_chunked()
    tokens_a, cp_a = lexer.lex_chunk(PYTHON_LINES[0], explicit)
    tokens_b, cp_b = lexer.lex_chunk(PYTHON_LINES[0])
    assert tokens_a == tokens_b
    assert cp_a == cp_b
