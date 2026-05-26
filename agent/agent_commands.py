"""Self-management commands the agent can embed in its replies.

The local model has no native tool calling, so it gives itself agency by
writing bracket commands such as ``[[LOCATION: office]]`` or
``[[NOTE: pick up milk]]`` anywhere in a reply. The app extracts and applies
these, then hides them from the user. ``CommandStreamFilter`` does the hiding
incrementally so commands never flash on screen while a reply streams in;
``parse_commands`` does the authoritative extraction once the reply is whole.
"""

import re

# Longest names first so the alternation prefers NOTE_DONE over NOTE.
COMMAND_KEYWORDS = ("LOCATION", "NOTE_DONE", "REMEMBER", "NOTE")

COMMAND_RE = re.compile(
    r"\[\[\s*(LOCATION|NOTE_DONE|REMEMBER|NOTE)\s*:\s*(.*?)\s*\]\]",
    re.IGNORECASE | re.DOTALL,
)


def parse_commands(text):
    """Strip command spans from a finished reply.

    Returns ``(clean_text, commands)`` where ``commands`` is a list of
    ``(KEYWORD, value)`` tuples in the order they appeared.
    """
    commands = []

    def _collect(match):
        commands.append((match.group(1).upper(), match.group(2).strip()))
        return ""

    clean = COMMAND_RE.sub(_collect, text)
    # Tidy whitespace left behind where a command used to sit.
    clean = re.sub(r"[ \t]{2,}", " ", clean)
    clean = re.sub(r" *\n", "\n", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    return clean.strip(), commands


class CommandStreamFilter:
    """Suppresses ``[[COMMAND: ...]]`` spans from streamed text.

    Feed it raw content deltas; it returns only the text that is safe to show.
    Text that might be the start of a command is held back until enough has
    arrived to decide. Unrelated ``[[...]]`` (not one of our keywords) passes
    through untouched.
    """

    def __init__(self):
        self._buf = ""

    def feed(self, chunk):
        """Add a streamed chunk; return text safe to display now."""
        self._buf += chunk
        return self._drain(final=False)

    def flush(self):
        """Stream is done; release anything still held back."""
        return self._drain(final=True)

    def _drain(self, final):
        out = []
        buf = self._buf
        while buf:
            start = buf.find("[[")
            if start == -1:
                # A lone trailing '[' could still grow into '[['; hold it.
                if not final and buf.endswith("["):
                    out.append(buf[:-1])
                    buf = "["
                else:
                    out.append(buf)
                    buf = ""
                break

            out.append(buf[:start])
            rest = buf[start:]
            end = rest.find("]]")

            if end == -1:
                inner = rest[2:].lstrip()
                if not final and self._could_be_command(inner):
                    buf = rest  # Hold; wait for the closing ]].
                    break
                # Not a command (or stream ended unclosed): emit the literal '[['.
                out.append("[[")
                buf = rest[2:]
                continue

            span = rest[: end + 2]
            if COMMAND_RE.match(span):
                buf = rest[end + 2:]  # Drop the command span entirely.
            else:
                out.append("[[")  # Some other bracketed text; keep it.
                buf = rest[2:]

        self._buf = buf
        return "".join(out)

    @staticmethod
    def _could_be_command(inner):
        """Could this not-yet-closed text after '[[' become one of our commands?"""
        head = re.split(r"[:\s]", inner, maxsplit=1)[0].upper()
        if head == "":
            return True
        return any(kw.startswith(head) or head.startswith(kw) for kw in COMMAND_KEYWORDS)
