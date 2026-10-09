"""Stable distinction between malformed input and unsupported semantics."""


class InvalidInput(ValueError):
    pass


class Unsupported(ValueError):
    pass
