import numpy as np
import pygame, operator
from abc import ABC, abstractmethod

GRAVITY_CONST = 9.81

datatype = np.float32

from typing import Callable, TypeVar, ParamSpec, Annotated
from numpy.typing import NDArray

T = TypeVar('T')
R = TypeVar('R')
P = ParamSpec('P')

class Point:
    coords: np.ndarray
    
    def __init__(self, coords: tuple[float, float] = (0, 0)) -> None:
        self.coords = np.array([*coords])

class Vector:
    dims: np.ndarray
    origins: np.ndarray
    len: int
    
    def __init__(self): 
        self.dims = np.array([0, 1])
        self.origins = np.array([0, 0]) # todo: into point
        self.len = 1
        
    def from_two_origins(self, from_vec: "Vector", to_vec: "Vector") -> "Vector":
        new_vec = Vector()
        
        new_vec.origins = from_vec.origins
        new_vec.dims = to_vec.origins - from_vec.origins # todo: reuse from_two_points
        
        return new_vec
    
    def from_two_points(self, from_p: Point, to_p: Point) -> "Vector":
        new_vec = Vector()

        new_vec.origins = from_p.coords
        new_vec.dims = to_p.coords - from_p.coords
        
        return new_vec

    def copy(self) -> "Vector":
        ret = Vector()
        ret.dims = self.dims.copy()
        ret.origins = self.origins.copy()
        ret.len = self.len
        return ret

    def get_scalar_len(self):
        return np.sqrt(np.square(self.dims).sum())

    def distance_to(self, point: Point): 
        # y = kx + b
        self_k = self.dims[1] / self.dims[0] # todo: catch zero division
        self_b = self.origins[1] + ( -self.origins[0] * self_k )
        
        other_k = -1/self_k
        other_b = point.coords[1] - other_k * point.coords[0]
        
        x = ( other_b - self_b ) / ( self_k - other_k )
        y = x * self_k + self_b
        
        len = Vector().from_two_points(Point(( x, y )), point).get_scalar_len()
        return len

class Transforms:
    """class provides basic vector transformations"""
    
    def rotate_ip(self, angle: float, transformant: Vector):
        transform_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                     [np.sin(angle), np.cos(angle)]])
        transformant.dims = transformant.dims @ transform_matrix.T
    
    def rotate(self, angle: float, transformant: Vector):
        new_vector = transformant.copy()
        self.rotate_ip(angle, new_vector)
        return new_vector

    def move_along(self, deltas: np.ndarray, transformant: Vector) -> Vector:
        new_vector = transformant.copy()
        angle = np.atan(new_vector.dims[0] / new_vector.dims[1])
        
        # new_vector.origins[0] += deltas[0] * np.sin(tilt) + deltas[1] * np.cos(tilt)
        # new_vector.origins[1] += deltas[0] * np.cos(tilt) + deltas[1] * np.sin(tilt)
        transform_matrix = np.array([[np.cos(angle), np.sin(angle)],
                                     [np.sin(angle), np.cos(angle)]])
        
        new_vector.origins = new_vector.origins + deltas @ transform_matrix
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
            
class Immovable:
    @abstractmethod
    def collides(self, other_p:Point) -> bool:
        pass
        
class Land(Immovable): 
    vpos: float
    def __init__(self) -> None:
        self.vpos = 0
        
    def collides(self, other_p:Point) -> bool: 
        if other_p.coords[1] <= self.vpos: return True
        return False
            
class RocketBody:
    dims: list[float]
    current_coordinates: np.ndarray 
    # main_vector: Vector
    
    def __init__(self, dims:tuple[float, float] = (5, 25), weight:float = 90, tilt: float = 0) -> None:
        self.dims = [dims[0], dims[1]]
        self.weight = weight
        self.tilt = tilt
        
        self.current_coordinates = np.array([0, 0], dtype=datatype)
        self.prev_coordinates = np.array([0, 0], dtype=datatype)
        self.current_vector = transforms.rotate(tilt, Vector())
        self.prev_vector = transforms.rotate(tilt, Vector())
        
    def get_mass_centre(self): 
        return self.current_vector.origins
    
    def add_tilt(self, angle): 
        self.tilt += angle
        transforms.rotate_ip(angle, self.current_vector)

class Rocket: 
    forces: list[Force]
    engines: list[Engine]
    hold_inplace: bool
    resulting_linear_acceleration: NDArray[datatype]

    def add_force(self, force: Force):
        self.forces.append(force)

    def __init__(self, forces: list[Force], engines: list[Engine], initial_tilt: float = 0):
        # delta movement of a rocket from prev step
        self.delta_coordinates = np.array([0, 0], dtype=datatype) 
        self.delta_angle = 0.
        self.current_linear_speed: np.ndarray = np.array([0, 0], dtype=datatype)
        self.current_rotation_speed: float = 0.
        self.prev_linear_speed = np.array([0, 0], dtype=datatype)
        self.prev_rotation_speed: float = 0.
        self.zero_vector = np.array([0, 0], dtype=datatype)
        self.resulting_linear_acceleration = np.array([0, 0], dtype=datatype)
        
        self.body_model = RocketBody(tilt = initial_tilt)

        self.prev_time = 0
        self.forces = forces
        self.engines = engines
        
        self.hold_inplace = True
        
    def release(self):
        self.hold_inplace = False

    def _time_step(self, time_step: float):
        self.prev_time += time_step

        self.prev_linear_speed: np.ndarray = self.current_linear_speed.copy()
        self.prev_rotation_speed: float = self.current_rotation_speed
        self.body_model.prev_coordinates = self.body_model.current_coordinates.copy()
        self.body_model.prev_vector = self.body_model.current_vector.copy()

        real_linear_acceleration = self.zero_vector.copy()
        real_rotation_acceleration: float = 0.

        for force in self.forces:
            arg_list: list[float] = []
            for parameter_name in force.attribute_names:
                arg_list.append(operator.attrgetter(parameter_name)(self))
            real_linear_acceleration = force.update(real_linear_acceleration, *arg_list) # type: ignore
        
        for engine in self.engines:
            engine_force_vector = engine.renew_force_vector(self.body_model.current_vector)
            engine_power = engine.current_force
            force_vector = engine_force_vector.dims * -1. * engine_power
            force_scalar = float(np.sqrt(force_vector[0] ** 2 + force_vector[1] ** 2))

            real_linear_acceleration += force_vector / self.body_model.weight
            
            rotation_moment = force_scalar * engine_force_vector.distance_to(Point((self.body_model.get_mass_centre()[0], self.body_model.get_mass_centre()[1])))
            inertia_moment = self.body_model.dims[0] * self.body_model.dims[1] * (self.body_model.dims[0] ** 2 + self.body_model.dims[1] ** 2) / 12
            
            real_rotation_acceleration += rotation_moment / inertia_moment
            
        self.resulting_linear_acceleration = real_linear_acceleration

        if not self.hold_inplace:
            self.current_linear_speed += real_linear_acceleration * time_step 
            self.current_rotation_speed += real_rotation_acceleration * time_step

            self.delta_coordinates = ( self.current_linear_speed * time_step + self.prev_linear_speed * time_step ) / 2
            self.delta_angle = ( self.current_rotation_speed * time_step + self.prev_rotation_speed * time_step ) / 2
            self.body_model.current_coordinates += self.delta_coordinates
            self.body_model.add_tilt(self.delta_angle)
            
    def revert_timestep_move(self):
        self.current_linear_speed = self.prev_linear_speed.copy()
        self.current_rotation_speed = self.prev_rotation_speed
        self.body_model.current_coordinates = self.body_model.prev_coordinates.copy()
        self.body_model.current_vector = self.body_model.prev_vector.copy()
        
class SimEnvironment: 
    rockets: NDArray[np.object_]
    land: Land
    def __init__(self, rockets: list[Rocket]) -> None:
        self.rockets = np.array(rockets, dtype=Rocket)
        self.land = Land()
        
    def time_step(self, time_step: float):
        for rocket in self.rockets: 
            rocket._time_step(time_step)
        
def init_rocket() -> SimEnvironment:

    main_engine_transforms: Callable[[Vector], Vector] = lambda v: transforms.rotate(np.pi, transforms.move_along(np.array([0, -10]), v))
    main_engine = Engine(200000, 58000, main_engine_transforms)

    rotating_engine_transforms: Callable[[Vector], Vector] = lambda v: transforms.rotate(np.pi / 2, transforms.move_along(np.array([0, -10]), v))
    rotating_engine = Engine(1000, 1000, rotating_engine_transforms)

    def gravity_func(force: np.ndarray, mass: float) -> np.ndarray:
        force[1] -= GRAVITY_CONST * mass
        return force
    gravity = Force(gravity_func, ["body_model.weight"])

    myrocket = Rocket(forces=[gravity], engines=[main_engine, rotating_engine], initial_tilt = 10 * (np.pi / 180))

    for engine in myrocket.engines:
        engine.turn_on()

    env = SimEnvironment([ myrocket ])
    return env


def launch_sim(screen: pygame.Surface, center_offset: tuple[int, int]):
    running = True
    dt = 0
    fps = 60
    
    env = init_rocket()
    
    myrocket = env.rockets[0]
    
    rocket_x = myrocket.body_model.current_coordinates[0] + center_offset[0]
    rocket_y = myrocket.body_model.current_coordinates[1] + center_offset[1]

    rocket_surface = pygame.Surface(myrocket.body_model.dims, pygame.SRCALPHA)
    rect = pygame.draw.rect(rocket_surface, "black", rocket_surface.get_rect())
    rect.center = (rocket_x, rocket_y)
    angle = myrocket.body_model.tilt * 180 / np.pi

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        dt = clock.tick(fps) / 1000 # todo
        env.time_step(dt)
        for rocket in env.rockets:
            if env.land.collides(Point(( rocket.body_model.current_coordinates[0], rocket.body_model.current_coordinates[1] ))):
                rocket.revert_timestep_move()                
        
        screen.fill("white")

        pygame.draw.rect(screen, "black", (0, center_offset[1] + myrocket.body_model.dims[1], center_offset[0] * 2, center_offset[1] * 2))
        rect.move_ip(*(myrocket.delta_coordinates.astype(int) * -1))
        old_center = rect.center

        angle = myrocket.body_model.tilt * 180 / np.pi
        rotated_surface = pygame.transform.rotate(rocket_surface, angle=angle * -1 % 360)
        rotated_rect = rotated_surface.get_rect()
        rotated_rect.center = old_center

        screen.blit(rotated_surface, rotated_rect)
        pygame.display.flip()
        
        for engine in myrocket.engines:
            engine.set_force(engine.current_force + 500)
            
        if myrocket.resulting_linear_acceleration[1] >= 0: 
            myrocket.release()

        print(myrocket.body_model.current_coordinates)

if __name__ == "__main__":
    pygame.init()
    infoObject = pygame.display.Info()
    screen = pygame.display.set_mode((infoObject.current_w / 2, infoObject.current_h / 2))
    clock = pygame.time.Clock()
    running = True
    
    center_offset = (int(infoObject.current_w / 4), int(infoObject.current_h / 4))

    launch_sim(screen, center_offset)
    
    pygame.quit()