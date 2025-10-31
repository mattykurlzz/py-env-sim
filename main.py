import numpy as np

GRAVITY_CONST = 9.81

datatype = np.float32

from typing import Callable, TypeVar, ParamSpec

T = TypeVar('T')
R = TypeVar('R')
P = ParamSpec('P')

class Vector:
    dims: np.ndarray
    origins: np.ndarray
    len: int
    
    def __init__(self): 
        self.dims = np.array([0, 1])
        self.origins = np.array([0, 0])
        self.len = 2

    def copy(self) -> "Vector":
        ret = Vector()
        ret.dims = self.dims.copy()
        ret.origins = self.origins.copy()
        ret.len = self.len
        return ret

class Transforms:
    """class provides basic vector transformations without changing given vector"""
    def rotate(self, angle: float, transformant: Vector):
        new_vector = transformant.copy()
        transform_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                     [np.sin(angle), np.cos(angle)]])
        new_vector.dims = new_vector.dims @ transform_matrix.T
        return new_vector
    
transforms = Transforms()

class Force:
    value_lambda: Callable[..., np.ndarray]
    attribute_names: list[str]
    def __init__(self, value_lambda: Callable[..., np.ndarray], attributes: list[str]):
        self.value_lambda = value_lambda
        self.attribute_names = attributes

    def update(self, *args) -> np.ndarray: # type: ignore
        return self.value_lambda(*args)

class Engine:
    maximum_force: float
    minimum_force: float
    current_force: float
    force_vector_transforms: Callable[[Vector], Vector]
    current_vector: Vector

    def __init__(self, max_force: float, min_force: float, force_vector_transforms: Callable[[Vector], Vector]):
        self.maximum_force = max_force
        self.minimum_force = min_force
        self.current_force = 0.
        self.force_vector_transforms = force_vector_transforms
        
    def renew_force_vector(self, main_vector: Vector) -> Vector:
        self.current_vector = self.force_vector_transforms(main_vector)
        return self.current_vector
    
    def turn_on(self):
        self.current_force = self.minimum_force
        
    def set_force(self, force: float):
        if force <= self.maximum_force and force >= self.minimum_force:
            self.current_force = force

class Rocket: 
    forces: list[Force]
    engines: list[Engine]

    def add_force(self, force: Force):
        self.forces.append(force)

    def __init__(self, mass: float, forces: list[Force], engines: list[Engine]):
        self.current_coordinates = np.array([0, 0], dtype=datatype)
        self.current_vector = Vector()
        self.current_speed: np.ndarray = np.array([0, 0], dtype=datatype)
        self.prev_speed = np.array([0, 0], dtype=datatype)
        self.current_force_applied = np.array([0, 0], dtype=datatype)

        self.rocket_mass = mass
        self.prev_time = 0
        self.forces = forces
        self.engines = engines

    def time_step(self, time: float):
        time_step = time - self.prev_time
        self.prev_time = time_step

        self.prev_speed: np.ndarray = self.current_speed.copy()
        real_acceleration = self.current_force_applied.copy()

        for force in self.forces:
            arg_list: list[float] = []
            for parameter_name in force.attribute_names:
                arg_list.append(getattr(self, parameter_name))
            real_acceleration = force.update(real_acceleration, *arg_list) # type: ignore
        
        for engine in self.engines:
            engine_force_vector = engine.renew_force_vector(self.current_vector)
            engine_power = engine.current_force
            real_acceleration += engine_force_vector.dims * -1. * engine_power

        self.current_speed += real_acceleration * time_step 
        self.current_coordinates += ( self.current_speed * time_step + self.prev_speed * time_step ) / 2


time = 0
time_step = 1

main_engine_transforms: Callable[[Vector], Vector] = lambda v: transforms.rotate(np.pi, v)
main_engine = Engine(2000, 100, main_engine_transforms)

def gravity_func(force: np.ndarray, mass: float) -> np.ndarray:
    force[1] -= GRAVITY_CONST * mass
    return force
gravity = Force(gravity_func, ["rocket_mass"])

myrocket = Rocket(mass=90, forces=[gravity], engines=[main_engine])

for engine in myrocket.engines:
    engine.turn_on()

for _ in range(100):
    time += time_step
    myrocket.time_step(time)
    
    for engine in myrocket.engines:
        engine.set_force(engine.current_force + 100)

    print(myrocket.current_coordinates)