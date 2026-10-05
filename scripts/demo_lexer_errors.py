#!/usr/bin/env python
"""
    Explicit lexer error reporting demo
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Demonstrates the error-reporting facility in :mod:`pygments.lexer`
    on deliberately bad source strings:

    * one-shot lexing records, for every ``Token.Error`` the lexer
      actually emits: the source line/column, the active (public) state
      and the byte range -- without a second lexing pass;
    * incremental, segmented lexing yields the exact same error positions
      as lexing the whole string at once, while the state and the absolute
      positions stay continuous across segments;
    * a lexer that raises an exception is attributed separately (with the
      failing stage) from ordinary error tokens.

    Run it with::

        python scripts/demo_lexer_errors.py

    :copyright: Copyright 2006-2026 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

import os
import sys

# always prefer Pygments from source if it exists
srcpath = os.path.join(os.path.dirname(__file__), '..')
if os.path.isdir(os.path.join(srcpath, 'pygments')):
    sys.path.insert(0, srcpath)

from pygments.lexer import RegexLexer, IncrementalLexer  # noqa: E402
from pygments.token import Text, Name, Whitespace       # noqa: E402


class DemoLexer(RegexLexer):
    """A toy language: words, whitespace and single-quoted strings."""
    tokens = {
        'root': [
            (r"'", Text, 'string'),
            (r'[A-Za-z_]+', Name),
            (r'\s+', Whitespace),
        ],
        'string': [
            (r"[A-Za-z ]+", Text),
            (r"'", Text, '#pop'),
        ],
    }


# A deliberately bad string: '$' characters are invalid everywhere and
# therefore become Token.Error.
BAD_SOURCE = "ok $$ word\nab $ tail\n"


def show(title, errors):
    print(title)
    for error in errors:
        if error.origin == 'token':
            print(f"  line {error.line}, col {error.column}  "
                  f"bytes [{error.start}:{error.end}]  "
                  f"state={error.state!r}  token={error.token_type}")
        else:  # origin == 'exception'
            print(f"  line {error.line}, col {error.column}  "
                  f"stage={error.stage!r}  error={error.error!r}")
    if not errors:
        print("  (no errors)")
    print()


def main():
    # 1) One-shot lexing.  The Error tokens stay in the token stream, and
    #    their locations are collected while that single pass runs.
    print("Source (later split along the first newline):")
    for number, line in enumerate(BAD_SOURCE.splitlines(), start=1):
        print(f"  {number}: {line}")
    print()

    one_shot = DemoLexer()
    list(one_shot.get_tokens(BAD_SOURCE))  # errors fill one_shot.errors
    show("One-shot lexing -- lexer.errors:", one_shot.errors)

    # 2) Incremental lexing.  Split at token boundaries (here: along the
    #    first newline) and finish() the last segment.
    split = BAD_SOURCE.index('\n') + 1
    incremental = IncrementalLexer(DemoLexer())
    list(incremental.feed(BAD_SOURCE[:split]))
    list(incremental.finish(BAD_SOURCE[split:]))
    show("Incremental lexing -- incremental.errors:", incremental.errors)

    same = [(e.start, e.end, e.line, e.column, e.state)
            for e in one_shot.errors] == \
           [(e.start, e.end, e.line, e.column, e.state)
            for e in incremental.errors]
    print("Error positions identical to the one-shot run:", same)


if __name__ == '__main__':
    main()
