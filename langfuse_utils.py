"""Optional Langfuse tracing.

If LANGFUSE_PUBLIC_KEY isn't set, get_langfuse_handler() returns None and
the app runs exactly as before — Langfuse is opt-in, not a hard dependency.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def get_langfuse_handler():
    """Return a Langfuse CallbackHandler if configured, else None.

    Sign up for the free Langfuse Cloud "Hobby" tier at
    https://cloud.langfuse.com, create a project, and copy the API keys
    into .env (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST).
    """
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        return None

    from langfuse.langchain import CallbackHandler

    return CallbackHandler()
