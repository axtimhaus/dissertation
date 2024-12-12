import deepdiff

from dissertation.sim.parameter_study.data_model import ParameterStudy


def hash(studies:list[ParameterStudy]) -> str:
    return deepdiff.DeepHash(studies)[studies]

STUDIES = [
    ParameterStudy(
        parameter_name='test',
        scale='log'
    )
]
