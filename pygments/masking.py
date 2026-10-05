"""
    pygments.masking
    ~~~~~~~~~~~~~~~~

    Mask (redact) character ranges of a highlighted token stream without
    re-running the lexer.

    Given a token stream as produced by ``Lexer.get_tokens`` (an iterable
    of ``(tokentype, value)`` pairs) and a set of absolute character
    ranges into the original source, :func:`mask_token_stream` replaces
    the covered text with a replacement string while:

    * keeping the token types of all (partially) outside tokens,
    * keeping absolute source offsets meaningful through an explicit
      output-to-source mapping,
    * never re-invoking the lexer,
    * never mutating the original token stream,
    * allowing the replacement to be shorter or longer than the masked
      source text,
    * refusing to split Unicode surrogate pairs,
    * failing atomically: invalid ranges raise :class:`MaskError` and no
      partially masked stream is produced.

    Range semantics
    ---------------
    A range is a ``(start, end)`` pair of absolute character offsets with
    ``0 <= start <= end <= len(source)``.  ``source[start:end]`` is
    replaced by the replacement string.  An *empty* range
    ``start == end`` masks zero source characters: the replacement is
    inserted at that position (pure insertion).  Ranges must not overlap;
    an empty range may share its position with the boundary of another
    range, in which case the insertion is applied first (ranges are
    processed in sorted order).

    Run the focused tests with::

        python -m pytest tests/test_masking.py -q

    :copyright: Copyright 2006-present by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

from typing import NamedTuple

from pygments.token import Token

__all__ = ['MaskError', 'MaskedRegion', 'MaskResult', 'mask_token_stream']


class MaskError(ValueError):
    """Raised when mask ranges are invalid.

    The error is raised before any masking output is produced, so a
    partially masked stream is never delivered.
    """


class MaskedRegion(NamedTuple):
    """Mapping of one masked region in the masked output text back to the
    original source.

    ``out_start``/``out_end`` are absolute character offsets into the
    masked output text; ``src_start``/``src_end`` are the absolute
    character offsets of the corresponding range in the original source.
    For an empty (insertion-only) range, ``src_start == src_end``.
    """

    out_start: int
    out_end: int
    src_start: int
    src_end: int


class MaskResult(NamedTuple):
    """Result of :func:`mask_token_stream`.

    ``tokens`` is the new masked token stream (a list of
    ``(tokentype, value)`` pairs; the input stream is left untouched),
    ``text`` is the masked source text, and ``mapping`` holds one
    :class:`MaskedRegion` per input range, in sorted range order.
    """

    tokens: list
    text: str
    mapping: list


def _is_high_surrogate(ch):
    return 0xD800 <= ord(ch) <= 0xDBFF


def _is_low_surrogate(ch):
    return 0xDC00 <= ord(ch) <= 0xDFFF


def _validate_ranges(ranges, length, source):
    """Normalize, sort and validate ranges; raise MaskError on any problem."""
    normalized = []
    for item in ranges:
        try:
            start, end = item
        except (TypeError, ValueError):
            raise MaskError(
                'range %r is not a (start, end) pair' % (item,)) from None
        if not isinstance(start, int) or not isinstance(end, int):
            raise MaskError(
                'range %r offsets must be integers' % (item,))
        if start < 0 or end < 0:
            raise MaskError(
                'range (%d, %d) has negative offsets' % (start, end))
        if start > end:
            raise MaskError(
                'range (%d, %d) has start > end' % (start, end))
        if end > length:
            raise MaskError(
                'range (%d, %d) exceeds source length %d'
                % (start, end, length))
        normalized.append((start, end))

    normalized.sort(key=lambda r: (r[0], r[1]))

    for prev, cur in zip(normalized, normalized[1:]):
        if cur[0] < prev[1]:
            raise MaskError(
                'ranges %r and %r overlap' % (prev, cur))

    for start, end in normalized:
        for pos in (start, end):
            if 0 < pos < length and _is_high_surrogate(source[pos - 1]) \
                    and _is_low_surrogate(source[pos]):
                raise MaskError(
                    'range (%d, %d) splits a surrogate pair at offset %d'
                    % (start, end, pos))
    return normalized


def mask_token_stream(tokens, ranges, replacement='*'):
    """Mask ``ranges`` of the source covered by a token stream.

    ``tokens`` is an iterable of ``(tokentype, value)`` pairs as produced
    by ``Lexer.get_tokens``; concatenating the values must yield the
    source text the ``ranges`` refer to.  ``ranges`` is an iterable of
    ``(start, end)`` absolute character offsets (see the module docstring
    for the exact semantics, including empty insertion ranges).
    ``replacement`` is the text substituted for each masked range (or
    inserted, for empty ranges); its length may differ from the masked
    source text.

    Returns a :class:`MaskResult`.  The input token stream is not
    modified, and the lexer is not re-run.  Raises :class:`MaskError`
    (before producing any output) if the ranges overlap, are out of
    bounds, or would split a Unicode surrogate pair.
    """
    if not isinstance(replacement, str):
        raise MaskError('replacement must be a string')

    token_list = list(tokens)
    source = ''.join(value for _, value in token_list)
    norm = _validate_ranges(ranges, len(source), source)

    n = len(norm)
    # out_spans[i] -> [out_start, out_end] accumulated for norm[i]
    out_spans = [[None, None] for _ in range(n)]

    out_tokens = []
    out_pos = 0          # absolute offset into the masked output text
    src_pos = 0          # absolute offset of the current token in source
    ri = 0               # first range index not fully consumed

    def emit_literal(text):
        nonlocal out_pos
        out_pos += len(text)

    def emit_replacement(idx):
        nonlocal out_pos
        if out_spans[idx][0] is None:
            out_spans[idx][0] = out_pos
        out_pos += len(replacement)
        out_spans[idx][1] = out_pos

    def apply_ranges(value, tok_start, relevant):
        """Build the masked value of one token from its relevant ranges."""
        parts = []
        cur = tok_start
        for idx in relevant:
            start, end = norm[idx]
            clip_start = max(start, tok_start)
            clip_end = min(end, tok_start + len(value))
            literal = source[cur:clip_start]
            parts.append(literal)
            emit_literal(literal)
            parts.append(replacement)
            emit_replacement(idx)
            cur = clip_end
        tail = source[cur:tok_start + len(value)]
        parts.append(tail)
        emit_literal(tail)
        return ''.join(parts)

    for ttype, value in token_list:
        tok_start = src_pos
        tok_end = src_pos + len(value)

        # Ranges relevant to this token: non-empty ranges intersecting
        # [tok_start, tok_end), plus empty ranges with
        # tok_start <= pos < tok_end.
        relevant = []
        j = ri
        while j < n and norm[j][0] < tok_end:
            start, end = norm[j]
            if end > tok_start or (start == end and start >= tok_start):
                relevant.append(j)
            j += 1

        if relevant:
            value = apply_ranges(value, tok_start, relevant)
        else:
            emit_literal(value)
        out_tokens.append((ttype, value))
        src_pos = tok_end

        # Advance the consumed pointer past the ranges this token fully
        # covers: non-empty ranges ending at or before tok_end are done,
        # and so are empty ranges strictly inside the token.  An empty
        # range sitting exactly at tok_end belongs to the next token (or
        # to the end-of-stream handling below).
        while ri < n and norm[ri][1] <= tok_end and \
                not (norm[ri][0] == norm[ri][1] and norm[ri][0] == tok_end):
            ri += 1

    # Leftover ranges can only be empty insertions at the very end of the
    # source (pos == len(source)); attach them to the last token, or to a
    # fresh Text token if the stream was empty.
    if ri < n:
        parts = []
        for idx in range(ri, n):
            start, end = norm[idx]
            if start != end or start != len(source):
                # Defensive: validation guarantees this cannot happen.
                raise MaskError(
                    'range (%d, %d) is not covered by the token stream'
                    % (start, end))
            parts.append(replacement)
            emit_replacement(idx)
        insertion = ''.join(parts)
        if out_tokens:
            ttype, value = out_tokens[-1]
            out_tokens[-1] = (ttype, value + insertion)
        else:
            out_tokens.append((Token.Text, insertion))

    mapping = [
        MaskedRegion(out_start, out_end, start, end)
        for (out_start, out_end), (start, end) in zip(out_spans, norm)
    ]
    text = ''.join(value for _, value in out_tokens)
    return MaskResult(tokens=out_tokens, text=text, mapping=mapping)
