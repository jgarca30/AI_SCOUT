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
    # Rol buscado (clave de config.ROLES del grupo del jugador) y, opcionalmente, pesos ajustados a mano
    rol: str = C.ROL_DEFECTO
    pesos_personalizados: dict[str, float] | None = None
    # Aplicar lo aprendido de las valoraciones del usuario (pesos aprendidos, Rocchio y descartes)
    usar_aprendizaje: bool = True
    ocultar_valorados: bool = False      # no volver a proponer candidatos ya valorados para esta referencia
    solo_mismo_estilo: bool = False      # restringir a jugadores del mismo estilo de juego que la referencia
    diversidad: float = 0.0              # 0 = solo similitud; hasta 0.9 = preselección más variada (MMR)
    # Club al que llegaría el sustituto: si se indica, el Analista de plantilla valora el encaje
    equipo_destino: str | None = None
    usar_critico: bool = True            # revisión de las fichas por el Jefe de scouting
    texto_original: str | None = None    # petición en lenguaje natural, si la consulta viene del Secretario técnico
    consulta_id: str = field(default_factory=nuevo_id_consulta)
    fecha_utc: str = field(default_factory=ahora_utc)

    def como_dict(self) -> dict:
        return asdict(self)
