"""Sample module for parser tests."""

import os
import json as j
from collections import OrderedDict
from typing import Optional as Opt

MAX_RETRIES = 3


# Computes the double of a number.
def double(value: int) -> int:
    """Return twice the given value."""
    inner_offset = 1

    def helper(x):
        return x + inner_offset

    return helper(value) * 2


class Greeter:
    """Greets people by name."""

    default_greeting = "Hello"

    def __init__(self, name: str) -> None:
        """Store the name to greet."""
        self.name = name

    def greet(self) -> str:
        """Build and return the greeting message."""
        return self._format(double(1))

    def _format(self, n: int) -> str:
        return f"{self.default_greeting}, {self.name}! ({n})"


class LoudGreeter(Greeter):
    """A Greeter that shouts."""

    def greet(self) -> str:
        return self._format(double(2)).upper()
