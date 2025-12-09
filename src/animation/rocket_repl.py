#!/usr/bin/env python3
import argparse
from os.path import exists
import shlex
import sys, os
import json
import cmd
import math
from typing import Callable
import numpy as np
import nats

import entities.core as entities
from entities.engine import Engine
from animation.core import Animation
from primitives.transforms import Transforms
from primitives.vector import Vector
from primitives.core import transforms


class RocketREPL(cmd.Cmd):
    intro = 'Simulation REPL. Type help or ? for commands. "quit" to exit.\n'
    prompt = '(sim) '
    simulation: entities.SimEnvironment | None = None
    animation: Animation | None = None

    vector_transforms: dict[str, list[Callable[[Vector], Vector]]] = {}
    engines_dict: dict[str, Engine] = {}
    timestamp = 0

    def __init__(self):
        super().__init__()
        return

    def do_init(self, arg=""):
        self.simulation = entities.SimEnvironment()
        print("Created empty simulation environment")

        self._update_animation()

    def do_add_rocket(self, arg):
        """add_rocket [id=<rocket_id>] [tilt=<initial_tilt>] [mass=<mass>]: Add rocket to simulation"""
        if self.simulation == None:
            raise AttributeError("simulation was not initialized")
        self.simulation.addRocketFromString(arg)

        self._update_animation()

    #def do_attach_engine(self, arg):
        #"""attach_engine <rocket_id> <power_min> [<power_max>]: Add engine """

    def do_create_vector_transforms(self, arg):
        """create_vector_transforms <name> [r=<rotation>=0] [origin_move=<x,y>=0,0]

        Add vector transforms to the simulation to create new vectors
        from a singular vector.

        Optional arguments:
          r             Rotation in degrees (default: 0)
          origin_move   Origin position adjustment as "x y" tuple (default: "0 0")
        """

        def rotate_factory(angle_deg: str):
            return lambda v: transforms.rotate(math.radians(float(angle_deg)), v)

        def move_along_factory(coords: str):
            x, y = tuple(map(float, coords.split(',')))
            return lambda v: transforms.move_along(np.array([x, y]), v)

        ALLOWED_ARGS = {'r' : rotate_factory, 'origin_move' : move_along_factory}
        parts = shlex.split(arg)

        params = {}
        for arg in parts:
            if '=' in arg:
                key, value = arg.split('=')
                if key in ALLOWED_ARGS.keys():
                    params[key] = ALLOWED_ARGS[key](value)

        transform_f: list[Callable[[Vector], Vector]] = [lambda v: v]
        for arg_key in list(params.keys()):  # Snapshot keys
            current_transform = params[arg_key]
            transform_f += [lambda v, t=current_transform: t(v)]

        self.vector_transforms[parts[0]] = transform_f

    def do_dislpay_vector_transforms(self, arg):
        """do_dislpay_vector_transforms: print all available vector transforms keys"""
        print(self.vector_transforms.keys())

    def do_create_engine(self, arg):
        """create_engine <name> <transforms-name> <minimal-force> [<maximum-force>]

        Add new engine that can me attached to a rocket

        Optional arguments:
          name              engine name that can be used to access it later
          transforms-name   name of transforms labmda that explains engine's force vector
          minimal-force     minimal force engine can have
          maximum-force     maximum force engine can have
        """
        parts = shlex.split(arg)

        if(not len(parts) >= 3):
            raise ValueError("function expects not less that 3 parameters")

        name = parts[0]
        transforms = self.vector_transforms[parts[1]]
        minimal_force = float(parts[2])
        maximum_force = minimal_force
        if len(parts) > 3:
             maximum_force = float(parts[3])

        self.engines_dict[name] = Engine(
            maximum_force, minimal_force, transforms)

    def do_dislpay_engines(self, arg):
        """do_dislpay_engines: print all available engine dict keys"""
        print(self.engines_dict.keys())

    def do_attach_engine(self, arg):
        """do_attach_engine <rocket-id> <engine-name>:
            attach created engine to a rocket"""
        if not self.simulation:
            raise AttributeError("simulation was not initialized")

        parts = shlex.split(arg)

        rocket_id = int(parts[0])
        engine_name = parts[1]
        engine = self.engines_dict[engine_name]

        self.simulation.addEngineById(rocket_id, (engine_name, engine))

    def do_turn_on_engine(self, arg):
        """do_turn_on_engine <rocket-id> <engine-name>:
            turn on engine attached to a rocket"""

        if not self.simulation:
            raise AttributeError("simulation was not initialized")

        parts = shlex.split(arg)

        rocket_id = int(parts[0])
        engine_name = parts[1]

        self.simulation.switchEngineById(rocket_id, engine_name)

    def do_modify_engine_force(self, arg):
        """do_modify_engine_force <rocket-id> <engine-name> <value>:
            change engine force value"""
        if not self.simulation:
            raise AttributeError("simulation was not initialized")

        parts = shlex.split(arg)
        rocket_id = int(parts[0])
        engine_name = parts[1]
        modify = float(parts[2])

        self.simulation.modifyEngineForceById(rocket_id, engine_name, modify)

    def do_set_engine_force(self, arg):
        """do_set_engine_force <rocket-id> <engine-name> <value>:
            change engine force value"""
        if not self.simulation:
            raise AttributeError("simulation was not initialized")

        parts = shlex.split(arg)
        rocket_id = int(parts[0])
        engine_name = parts[1]
        modify = float(parts[2])

        self.simulation.setEngineForceById(rocket_id, engine_name, modify)

    def do_step(self, arg):
        """step <step_s>=1: Move simualtion further"""
        if self.simulation == None:
            raise AttributeError("simulation was not initialized")
        self.timestamp += float(arg)
        self.simulation.time_step(arg)

        self._update_animation()
        
    def _get_timestamp(self) -> float:
        return self.timestamp

    def do_animation_init(self, arg=""):
        """animation_init [<fps>]"""
        if arg != "":
            self.animation = Animation(arg)
        else:
            self.animation = Animation()

        self._update_animation()

    def do_release_rocket(self, arg):
        """release_rocket <rocket_id>: Release rockets and make them affected by simulation environment"""
        if not self.simulation:
            raise AttributeError("simulation was not initialized")

        self.simulation.releaseRoket(arg)

    def do_EOF(self, arg):
        if type(self.animation) == Animation:
            del self.animation
        sys.exit(0)
        
    def do_get_rocket_state(self, arg) -> dict[str, dict[str, float]]:
        """get_rocket_state <rocket-id>: Return json-type string with current rocket state"""
        if not self.simulation:
            raise AttributeError("simulation was not initialized")
            
        id = int(arg)
        
        rocket_used = None
        for rocket in self.simulation.rockets:
            if rocket.id == id: 
                rocket_used = rocket
                break
        
        if rocket_used:
            state_dict = rocket_used.get_state_dict()
            return state_dict
        return {}

    def _update_animation(self):
        if self.animation and self.simulation:
            self.animation.update(self.simulation)

def main():
    RocketREPL().cmdloop()

if __name__ == "__main__":
    main()
