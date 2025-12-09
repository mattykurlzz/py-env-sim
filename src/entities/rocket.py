import numpy as np
import operator
from numpy.typing import NDArray
import shlex
import math, json

from primitives.core import transforms, Vector, Force, Point
from entities.engine import Engine

class RocketBody:
    dims: list[float]
    current_coordinates: np.ndarray
    # main_vector: Vector

    def __init__(
        self, dims: tuple[float, float] = (5, 25), mass: float = 90, tilt: float = 0
    ) -> None:
        self.dims = [dims[0], dims[1]]
        self.mass = mass
        self.tilt = tilt

        self.current_coordinates = np.array([0, 0], dtype=np.float128)
        self.prev_coordinates = np.array([0, 0], dtype=np.float128)
        self.current_vector = transforms.rotate(tilt, Vector())
        self.prev_vector = transforms.rotate(tilt, Vector())

    def get_mass_centre(self):
        return self.current_vector.origins

    def add_tilt(self, angle):
        self.tilt += angle
        transforms.rotate_ip(angle, self.current_vector)

class Rocket:
    id: int

    forces: list[Force]
    engines: dict[str, Engine]

    hold_inplace: bool
    resulting_linear_acceleration: NDArray[np.float128]
    body_model: RocketBody

    ALLOWED_PARAMS = {'initial_tilt'}
    
    def get_state_dict(self):
        ret_dict = {
            "coordinates": {
                "x": float(self.body_model.current_coordinates[0]),
                "y": float(self.body_model.current_coordinates[1]),
                "rotation": float(self.body_model.tilt)
            },
            "velocity": {
                "x": float(self.current_linear_speed[0]),
                "y": float(self.current_linear_speed[1]),
                "rotation": float(self.current_rotation_speed),
            },
            "acceleration": {
                "x": float(self.real_linear_acceleration[0]),
                "y": float(self.real_linear_acceleration[1]),
                "rotation": float(self.real_rotation_acceleration)
            },
        }
        return ret_dict

    def add_force(self, force: Force):
        self.forces.append(force)

    def add_engine(self, engine_tuple: tuple[str, Engine]):
        self.engines[engine_tuple[0]] = engine_tuple[1]

    def switch_engine(self, engine_name: str):
        self.engines[engine_name].turn_on()

    def modify_engine_force(self, engine_name: str, mod_value: float):
        self.engines[engine_name].current_force = self.engines[engine_name].current_force + mod_value

    def set_engine_force(self, engine_name: str, mod_value: float):
        self.engines[engine_name].current_force = mod_value

    def __init__(
        self, id: int, forces: list[Force], mass: float = 900 , tilt: float = 0
    ):
        self.id = id
        # delta movement of a rocket from prev step
        self.delta_coordinates = np.array([0, 0], dtype=np.float128)
        self.delta_angle = 0.0
        self.current_linear_speed: np.ndarray = np.array([0, 0], dtype=np.float128)
        self.current_rotation_speed: float = 0.0
        self.prev_linear_speed = np.array([0, 0], dtype=np.float128)
        self.prev_rotation_speed: float = 0.0
        self.zero_vector = np.array([0, 0], dtype=np.float128)
        self.resulting_linear_acceleration = np.array([0, 0], dtype=np.float128)
        self.real_linear_acceleration = np.array([0, 0], dtype=np.float128)
        self.real_rotation_acceleration: float = 0.0
        self.prev_time = 0

        self.body_model = RocketBody(tilt=math.radians(tilt))

        self.forces = forces
        self.engines = {}

        self.hold_inplace = True

    def release(self):
        self.hold_inplace = False

    def _time_step(self, time_step: float):
        self.prev_time += time_step

        self.prev_linear_speed: np.ndarray = self.current_linear_speed.copy()
        self.prev_rotation_speed: float = self.current_rotation_speed
        self.body_model.prev_coordinates = self.body_model.current_coordinates.copy()
        self.body_model.prev_vector = self.body_model.current_vector.copy()

        self.real_linear_acceleration = self.zero_vector.copy()
        self.real_rotation_acceleration: float = 0.0

        for force in self.forces:
            arg_list: list[float] = []
            for parameter_name in force.attribute_names:
                arg_list.append(operator.attrgetter(parameter_name)(self))
            self.real_linear_acceleration = force.update(self.real_linear_acceleration, *arg_list) / self.body_model.mass  # type: ignore

        for engine in self.engines.values():
            engine_force_vector = engine.renew_force_vector(
                self.body_model.current_vector
            )
            engine_power = engine.current_force
            force_vector = engine_force_vector.dims * -1.0 * engine_power
            force_scalar = float(np.sqrt(force_vector[0] ** 2 + force_vector[1] ** 2))

            self.real_linear_acceleration += force_vector / self.body_model.mass

            rotation_moment = force_scalar * engine_force_vector.distance_to(
                Point(
                    (
                        self.body_model.get_mass_centre()[0],
                        self.body_model.get_mass_centre()[1],
                    )
                )
            )
            inertia_moment = (
                self.body_model.dims[0]
                * self.body_model.dims[1]
                * (self.body_model.dims[0] ** 2 + self.body_model.dims[1] ** 2)
                / 12
            )

            self.real_rotation_acceleration += rotation_moment / inertia_moment

        self.resulting_linear_acceleration = self.real_linear_acceleration

        if not self.hold_inplace:
            self.current_linear_speed += self.real_linear_acceleration * time_step
            self.current_rotation_speed += self.real_rotation_acceleration * time_step

            self.delta_coordinates = (
                self.current_linear_speed * time_step
                + self.prev_linear_speed * time_step
            ) / 2
            self.delta_angle = (
                self.current_rotation_speed * time_step
                + self.prev_rotation_speed * time_step
            ) / 2
            self.body_model.current_coordinates += self.delta_coordinates
            self.body_model.add_tilt(self.delta_angle)

    def revert_timestep_move(self):
        self.current_linear_speed = self.prev_linear_speed.copy()
        self.current_rotation_speed = self.prev_rotation_speed
        self.body_model.current_coordinates = self.body_model.prev_coordinates.copy()
        self.body_model.current_vector = self.body_model.prev_vector.copy()