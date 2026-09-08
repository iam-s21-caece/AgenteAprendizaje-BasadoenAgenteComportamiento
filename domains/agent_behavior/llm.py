"""Cliente LLM mínimo para el dominio (DeepSeek / API compatible con OpenAI).

Dos decisiones que explican por qué este archivo se ve así
-----------------------------------------------------------
1. **Solo librería estándar.** El proyecto no tiene dependencias externas y eso vale la pena
   conservarlo: la imagen Docker no necesita `pip install`, el arranque no depende de PyPI y no
   hay superficie de versiones que se rompa. La API es HTTP+JSON, así que `urllib.request`
   alcanza. Si algún día se quiere el SDK oficial, se cambia esta clase y nada más.

2. **Vive en el dominio, nunca en el núcleo.** El núcleo no sabe que existe un LLM, igual que
   no sabe que existen las trazas. Si mañana el LLM se saca, el agente sigue aprendiendo — el
   aprendizaje no depende de él, y ese es el punto.

Regla de operación: este cliente NUNCA propaga excepciones. Un timeout, una cuota agotada o una
respuesta rara devuelven `None` y el agente sigue con su heurística. El LLM es un acompañante
del razonamiento, no un punto único de falla del loop.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional


class DeepSeekClient:
    """Cliente de chat completions sobre `urllib` (stdlib).

    Args:
        api_key: credencial; si viene vacía el cliente queda deshabilitado y devuelve `None`.
        base_url: raíz de la API.
        model: modelo a invocar.
        temperature: baja por defecto — se le pide análisis sobre evidencia, no creatividad.
        timeout_seconds: tope duro; el loop del agente no puede quedarse esperando a una API.
        max_tokens: techo de respuesta, para que una sugerencia de exploración no se vuelva un
            ensayo ni un gasto imprevisto.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        temperature: float = 0.2,
        timeout_seconds: int = 45,
        max_tokens: int = 500,
    ) -> None:
        self._api_key = (api_key or "").strip()
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._timeout = timeout_seconds
        self._max_tokens = max_tokens

    @property
    def enabled(self) -> bool:
        """`False` si no hay credencial: el dominio decide degradar sin romperse."""
        return bool(self._api_key)

    def complete(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Pide una respuesta al modelo.

        Returns:
            El texto de la respuesta, o `None` ante cualquier problema (sin credencial, error
            de red, timeout, respuesta inesperada). El llamador debe tener siempre un plan B.
        """
        if not self.enabled:
            return None

        body = json.dumps(
            {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
                "stream": False,
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            choices = payload.get("choices") or []
            if not choices:
                return None
            content = (choices[0].get("message") or {}).get("content")
            return content.strip() if isinstance(content, str) and content.strip() else None
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, OSError):
            # Deliberadamente amplio: cualquier fallo del LLM degrada a heurística.
            return None
