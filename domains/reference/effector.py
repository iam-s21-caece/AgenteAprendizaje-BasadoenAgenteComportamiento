"""Implementación de referencia del Effector.

La acción externa del agente en esta familia es 'emitir/registrar la alerta'. Aquí se
materializa como acumular la alerta en memoria (y, opcionalmente, imprimirla). Nada de
integraciones externas a propósito: el punto del dominio de referencia es ser el molde mínimo,
no un conector real. Un dominio productivo reemplazaría esto por un ticket, un mail, etc.,
respetando el mismo contrato `Effector.act`."""

from __future__ import annotations

from typing import List

from core import Effector, Prediction


class ReferenceEffector(Effector):
    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose
        # Registro de alertas efectivamente emitidas: útil para inspección y para el experimento.
        self.emitted: List[Prediction] = []

    def act(self, prediction: Prediction) -> None:
        self.emitted.append(prediction)
        if self._verbose:
            print(
                f"  [ALERTA] entidad={prediction.entity_id} t={prediction.timestamp} "
                f"desvio={prediction.deviation_sigmas:+.2f} sigmas (ciclo {prediction.cycle})"
            )
