=====================
The full Pygments API
=====================

This page describes the Pygments API.

High-level API
==============

.. module:: pygments

Functions from the :mod:`pygments` module:

.. autofunction:: lex
.. autofunction:: format
.. autofunction:: highlight


.. module:: pygments.lexers

Functions from :mod:`pygments.lexers`:

.. autofunction:: get_lexer_by_name
.. autofunction:: get_lexer_for_filename
.. autofunction:: get_lexer_for_mimetype
.. autofunction:: load_lexer_from_file
.. autofunction:: guess_lexer
.. autofunction:: guess_lexer_for_filename
.. autofunction:: get_all_lexers
.. autofunction:: find_lexer_class_by_name
.. autofunction:: find_lexer_class


.. module:: pygments.formatters

Functions from :mod:`pygments.formatters`:

.. autofunction:: get_formatter_by_name
.. autofunction:: get_formatter_for_filename
.. autofunction:: load_formatter_from_file


.. module:: pygments.styles

Functions from :mod:`pygments.styles`:

.. autofunction:: get_style_by_name
.. autofunction:: get_all_styles
.. autodata:: STYLE_MAP

.. module:: pygments.lexer

Lexers
======

The base lexer class from which all lexers are derived is:

.. autoclass:: Lexer
   :members: __init__, add_filter, get_tokens, get_tokens_unprocessed, analyse_text

There are several base classes derived from ``Lexer`` you can use to build your
lexer from:

.. autoclass:: pygments.lexer.RegexLexer
.. autoclass:: pygments.lexer.ExtendedRegexLexer
.. autoclass:: pygments.lexer.DelegatingLexer


Chunked (incremental) lexing
============================

Applications like log viewers and editors often cannot re-lex the whole
document on every change.  For them, `pygments.lexer.RegexLexer` offers an
explicit chunked interface: you feed the source one piece at a time and carry
an opaque `~pygments.lexer.LexerCheckpoint` between calls.

.. automethod:: pygments.lexer.RegexLexer.lex_chunk
.. automethod:: pygments.lexer.RegexLexer.start_chunked
.. autoclass:: pygments.lexer.LexerCheckpoint

The following exceptions are raised by the chunked interface:

.. autoexception:: pygments.lexer.CheckpointError
.. autoexception:: pygments.lexer.CheckpointMismatchError
.. autoexception:: pygments.lexer.CheckpointUnsupportedError

Rules and guarantees:

* Chunks must be cut at explicit line boundaries (every chunk except the
  last one must end with ``'\n'``); the last chunk may be unterminated if
  ``final=True`` is passed.
* Multi-line strings, comments and nested states are carried over via the
  checkpoint's state stack, and token offsets are absolute -- they are
  never reset between chunks.
* A token spanning a chunk boundary is emitted once per chunk; merging
  adjacent tokens of the same type yields exactly the token stream of
  one-shot lexing.
* A checkpoint is bound to the lexer class *and* its options.  Resuming it
  with a different lexer or different options raises
  `~pygments.lexer.CheckpointMismatchError`.  Checkpoints are picklable
  and contain no source text, so they stay small.

A lexer supports checkpointing if all of the following hold:

* it is a plain `~pygments.lexer.RegexLexer` subclass (not
  `~pygments.lexer.ExtendedRegexLexer` or
  `~pygments.lexer.DelegatingLexer`),
* it does not override ``get_tokens_unprocessed()`` with custom logic
  (lexers doing so are rejected with
  `~pygments.lexer.CheckpointUnsupportedError`),
* it does not set ``supports_checkpoints = False`` (lexers whose token
  actions keep hidden state in instance attributes should set this),
* its multi-line constructs (strings, comments) are modeled with lexer
  *states*, as is the case for the built-in Python, Rust and many other
  lexers.  Rules whose regular expressions themselves match across line
  boundaries (e.g. line-continuation handling or single-regex multi-line
  comments) are outside the equivalence guarantee.

Short demo:

.. code-block:: python

    from pygments.lexers import PythonLexer

    lines = ['x = """hello\n', 'world"""\n', 'print(x)\n']

    lexer = PythonLexer()
    checkpoint = None
    for i, line in enumerate(lines):
        tokens, checkpoint = lexer.lex_chunk(line, checkpoint)
        # tokens: list of (absolute_offset, tokentype, value)
        print(tokens)

    # A checkpoint can be saved and resumed by a fresh lexer instance:
    lexer_a = PythonLexer()
    _, checkpoint = lexer_a.lex_chunk(lines[0])
    lexer_b = PythonLexer()
    tokens, checkpoint = lexer_b.lex_chunk(lines[1], checkpoint)


.. module:: pygments.formatter

Formatters
==========

A formatter is derived from this class:


.. autoclass:: Formatter
   :members: __init__, get_style_defs, format


.. module:: pygments.util

Utilities
=========

The :mod:`pygments.util` module has some utility functions usable for processing
command line options. All of the following functions get values from a
dictionary of options. If the value is already in the type expected by the
option, it is returned as-is. Otherwise, if the value is a string, it is first
converted to the expected type if possible.

.. autoexception:: OptionError
.. autofunction:: get_bool_opt
.. autofunction:: get_int_opt
.. autofunction:: get_list_opt
.. autofunction:: get_choice_opt

It also defines an exception:

.. autoexception:: ClassNotFound
