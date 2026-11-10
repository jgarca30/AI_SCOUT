"""Estructuras de datos compartidas por los módulos."""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from . import config as C


def ahora_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def nuevo_id_consulta() -> str:
    return datetime.now(timezone.utc).strftime("Q%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


@dataclass
class Consulta:
    """Parámetros de una búsqueda de sustituto (contrato común de consulta)."""

    objetivo_registro_id: str
    ligas_permitidas: list[str] = field(default_factory=lambda: list(C.LIGAS_NORMALIZADAS))
    minutos_minimos: int = C.MINUTOS_MINIMOS_DEFECTO
    edad_maxima: int | None = None
    usar_llm: bool = True
    top_preseleccion: int = C.TOP_PRESELECCION
    top_final: int = C.TOP_FINAL
    peso_numerico: float = C.PESO_NUMERICO
    peso_cualitativo: float = C.PESO_CUALITATIVO
    consulta_id: str = field(default_factory=nuevo_id_consulta)
    fecha_utc: str = field(default_factory=ahora_utc)

    def como_dict(self) -> dict:
        return asdict(self)
