from typing import Literal

from pydantic import BaseModel, ConfigDict


class ParameterStudy(BaseModel):
    model_config = ConfigDict(frozen=True)

    parameter_name: str
    min : float = 0
    max: float = 10
    count: int = 10
    scale: Literal["lin", "log", "geom"]
    display_tex: str = ""

    def __str__(self) -> str:
        return f"{self.parameter_name}_{self.min}_{self.max}_{self.scale}_{self.count}"
