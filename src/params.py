from enum import Enum
from typing import List

from pydantic import BaseModel, model_validator


class RepopulateStrategy(Enum):
    RANDOM_REPLACE = 1


class GradientParams(BaseModel):
    num_attractors: int
    num_repellers: int
    gradient_strength: float
    pole_motion: float
    standard_deviation: float


class ParticleParams(BaseModel):
    num_particle_types: int
    num_initial_particles: int
    initial_distribution: List[float]
    repopulate_strategy: RepopulateStrategy
    bond_matrix: List[List[float]]
    attraction_matrix: List[List[float]]


class InteractionParams(BaseModel):
    brownian_motion: float
    # TODO: can have different interaction distances/strengths
    max_interaction_distance: float
    repulsion_distance: float
    repulsion_strength: float
    bond_strength: float
    max_motion: float


class GridParams(BaseModel):
    max_x: int
    max_y: int
    grid_size: int


class Params(BaseModel):
    grid: GridParams
    interaction: InteractionParams
    particle: ParticleParams
    gradient: GradientParams

    # TODO: probably unnecessary, need to refactor grid
    @model_validator(mode='after')
    def validate_grid_size(self):
        assert self.grid.grid_size >= 2 * \
            self.interaction.max_interaction_distance
        return self
