"""Capa transversal: deduplicación por firma con cooldown.

Por qué existe
--------------
El performance element puede querer alertar sobre la misma entidad en ciclos consecutivos
mientras dura un desvío. Emitir la misma alerta una y otra vez es ruido que degrada la utilidad
del agente (fatiga de alertas) sin aportar información nueva. Esta capa garantiza 'una alerta por
firma dentro de una ventana de enfriamiento'. Vive en el núcleo porque protege la calidad de la
ACCIÓN del loop, independientemente del dominio; qué constituye la 'firma' lo define el dominio
al construir la `Decision`.
"""

from __future__ import annotations

from typing import Dict


class SignatureDeduper:
    """Suprime alertas repetidas por firma durante `cooldown` ciclos.

    No decide si algo es una alerta (eso es del performance element); decide si una alerta ya
    decidida debe EMITIRSE ahora o callarse por repetida. Mantener esa frontera evita que el
    dedupe oculte información al aprendizaje: la predicción igual se registra y se juzga; solo
    se evita el efecto externo repetido."""

    def __init__(self, cooldown: int = 5) -> None:
        # cooldown en ciclos: cuánto silencio se exige entre dos emisiones de la misma firma.
        self._cooldown = cooldown
        self._last_emitted: Dict[str, int] = {}

    def should_emit(self, signature: str, cycle: int) -> bool:
        """True si esta firma no se emitió dentro de la ventana de cooldown. Registra la emisión."""
        last = self._last_emitted.get(signature)
        if last is not None and (cycle - last) < self._cooldown:
            return False
        self._last_emitted[signature] = cycle
        return True
