from pydantic import BaseModel, ConfigDict, Field


class TelemetryPacket(BaseModel):
    """Contrato canónico de telemetría F1 (Pydantic v2).

    Regla .clinerules #1: todo intercambio entre módulos usa este esquema.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    timestamp_ms: float = Field(ge=0, description="Tiempo de sesión en milisegundos")
    lap_number: int = Field(ge=1, description="Número de vuelta (1-based)")
    x: float = Field(ge=0, le=1000, description="Coordenada X normalizada [0, 1000]")
    y: float = Field(ge=0, le=1000, description="Coordenada Y normalizada [0, 1000]")
    speed_kmh: float = Field(ge=0, le=450, description="Velocidad en km/h")
    throttle_pct: float = Field(ge=0, le=100, description="Acelerador en %")
    brake_pct: float = Field(ge=0, le=100, description="Freno en % (0 o 100 si es binario)")
    gear: int = Field(ge=0, le=8, description="Marcha engranada")
    rpm: int = Field(ge=0, le=20000, description="Revoluciones por minuto")
    drs_active: bool = Field(description="True si el DRS está abierto")
    tyre_compound: str = Field(min_length=1, description="Compuesto, p.ej. SOFT/MEDIUM/HARD")
    tyre_age_laps: int = Field(ge=0, description="Edad del neumático en vueltas")
    pinn_tyre_temp_c: float | None = Field(default=None, ge=-50, le=200)
    pinn_brake_temp_c: float | None = Field(default=None, ge=-50, le=1500)
    stochastic_grip_index: float | None = Field(default=None, ge=0, le=2)
