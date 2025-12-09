#!/usr/bin/env python3
from animation.rocket_repl import RocketREPL
import pygame

def main():
    clock = pygame.time.Clock()
    elapsed_t = 0

    repl = RocketREPL()

    repl.do_init()
    repl.do_animation_init()

    repl.do_add_rocket("id=1 tilt=45 mass=1500")
    repl.do_create_vector_transforms("main_vector origin_move=0,-10 r=180")
    repl.do_create_vector_transforms("control_1 origin_move=1,-10 r=160")
    repl.do_create_vector_transforms("control_2 origin_move=-1,-8 r=200")

    repl.do_create_engine("main_engine main_vector 1500 2000")
    repl.do_create_engine("control_engine_1 control_1 0 1000")
    repl.do_create_engine("control_engine_2 control_2 0 1000")

    repl.do_attach_engine("1 main_engine")
    repl.do_attach_engine("1 control_engine_1")
    repl.do_attach_engine("1 control_engine_2")

    repl.do_turn_on_engine("1 main_engine")
    repl.do_turn_on_engine("1 control_engine_1")
    repl.do_turn_on_engine("1 control_engine_2")

    current_force = 10000
    released = False
    control_started_clock = False
    control_ended_clock = False
    control_started_counterclock = False
    control_ended_counterclock = False
    prev_elapsed = 0
    while(True):
        dt = clock.tick(60) / 1000
        repl.do_step(f"{dt}")
        elapsed_t += dt
        
        print(repl.do_get_rocket_state("1")["coordinates"])

        if prev_elapsed < elapsed_t - 1:
            prev_elapsed = elapsed_t
            repl.do_modify_engine_force("1 main_engine 10")
            current_force += 10

            if current_force >= 7000 and not released:
                repl.do_release_rocket("1")
                released = True
                
            if elapsed_t > 10 and not control_started_clock: 
                print("started control engine 1")
                repl.do_modify_engine_force("1 control_engine_1 50")
                control_started_clock = True
                
            if elapsed_t > 12 and not control_ended_clock:
                print("halted control engine 1")
                repl.do_modify_engine_force("1 control_engine_1 -50")
                control_ended_clock = True

            if elapsed_t > 20 and not control_started_counterclock: 
                print("started control engine 2")
                repl.do_modify_engine_force("1 control_engine_2 50")
                control_started_counterclock = True
                
            if elapsed_t > 25 and not control_ended_counterclock:
                print("halted control engine 2")
                repl.do_modify_engine_force("1 control_engine_2 -50")
                control_ended_counterclock = True

if __name__ == "__main__":
    main()
