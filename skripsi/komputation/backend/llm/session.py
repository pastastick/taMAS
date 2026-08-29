"""session.py — sesi chat multi-turn di atas LocalLLMBackend.

Diekstrak dari bekas `client.py` BAGIAN 8. Riwayat percakapan disimpan di
memory (bukan di KV-cache) — pembungkus tipis untuk kompatibilitas dengan
`ChatSession` lama, bukan jalur produksi (yang memakai `run()`/KV langsung).

Type hint `backend: LocalLLMBackend` sengaja forward-reference (`TYPE_CHECKING`
saja) — `LocalLLMBackend` (di `backend.py`) yang membangun `LocalChatSession`
lewat `build_chat_session()`, jadi impor penuh di sini akan siklik. Nilai
`DEFAULT_SYSTEM_PROMPT` diambil lewat instance (`backend.DEFAULT_SYSTEM_PROMPT`)
alih-alih lewat kelas, persis karena alasan yang sama.
"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from llm.backend import LocalLLMBackend


class LocalChatSession:
    """
    Multi-turn session. History di memory.
    Compatible dengan ChatSession di client.py.
    """

    def __init__(
        self,
        backend        : "LocalLLMBackend",
        conversation_id: Optional[str] = None,
        system_prompt  : Optional[str] = None,
    ) -> None:
        self.backend         = backend
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.system_prompt   = system_prompt or backend.DEFAULT_SYSTEM_PROMPT
        self._history: List[Dict[str, str]] = []

    def build_chat_completion(self, user_prompt: str, **kwargs) -> str:
        resp = self.backend.build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=self.system_prompt,
            former_messages=self._history,
            **kwargs,
        )
        self._history.append({"role": "user",      "content": user_prompt})
        self._history.append({"role": "assistant", "content": resp})
        return resp

    def get_conversation_id(self) -> str:
        return self.conversation_id

    def clear_history(self) -> None:
        self._history = []
