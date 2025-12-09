from entities.immovable.core import *
from entities.engine import *
from entities.rocket import *
from entities.immovable.land import Land

GRAVITY_CONST = 9.81

class SimEnvironment:
    rockets: list[Rocket]
    land: Land

    state: dict

    def __init__(self, rockets: list[Rocket] = []) -> None:
        self.rockets = rockets
        self.land = Land()

        self.state = {}

    def time_step(self, time_step: float | str):
        time_step = float(time_step)

        for rocket in self.rockets:
            rocket._time_step(time_step)

    def addRocketFromString(self, arg: str):
        ALLOWED_ARGS = {'id' : int, 'tilt' : float, 'mass': float}
        parts = shlex.split(arg)

        params = {}
        for arg in parts:
            if '=' in arg:
                key, value = arg.split('=', 1)
                if key in ALLOWED_ARGS.keys():
                    params[key] = ALLOWED_ARGS[key](value)
        if id not in params.keys():
            max_id = 0
            for rocket in self.rockets:
                if rocket.id > max_id: max_id = rocket.id
            max_id += 1
            params['id'] = max_id
        else:
            for rocket in self.rockets:
                if rocket.id == params['id']:
                    raise ValueError(f"rocket with id {rocket.id} already exists")

        def gravity_func(force: np.ndarray, mass: float) -> np.ndarray:
            force[1] -= GRAVITY_CONST * mass
            return force

        gravity = Force(gravity_func, ["body_model.mass"])

        self.rockets.append(Rocket(**params, forces=[gravity]))

    def releaseRoket(self, id: int | str):
        id = int(id)
        
        for rocket in self.rockets:
            if rocket.id == id: rocket.release()

    def addEngineById(self, id: int, engine_tuple: tuple[str, Engine]):
        engined_rocket = None
        for rocket in self.rockets:
            if id == rocket.id: 
                engined_rocket = rocket
                break

        if not engined_rocket:
            raise ValueError(f"No rocket with id {id} found")
        
        engined_rocket.add_engine(engine_tuple)

    def switchEngineById(self, id: int, engine_name: str):
        engined_rocket = None
        for rocket in self.rockets:
            if id == rocket.id: 
                engined_rocket = rocket
                break

        if not engined_rocket:
            raise ValueError(f"No rocket with id {id} found")
        
        engined_rocket.switch_engine(engine_name)

    def modifyEngineForceById(self, id: int, engine_name: str, engine_force_mod: float):
        engined_rocket = None
        for rocket in self.rockets:
            if id == rocket.id: 
                engined_rocket = rocket
                break

        if not engined_rocket:
            raise ValueError(f"No rocket with id {id} found")
        
        engined_rocket.modify_engine_force(engine_name, engine_force_mod)

    def setEngineForceById(self, id: int, engine_name: str, engine_force_mod: float):
        engined_rocket = None
        for rocket in self.rockets:
            if id == rocket.id: 
                engined_rocket = rocket
                break

        if not engined_rocket:
            raise ValueError(f"No rocket with id {id} found")
        
        engined_rocket.set_engine_force(engine_name, engine_force_mod)