#!/usr/bin/env python3
from animation.rocket_repl import RocketREPL
import pygame
import asyncio
import nats
from nats.errors import TimeoutError
import json

TICK = 0.001

repl = RocketREPL()
clock = pygame.time.Clock()

async def send_timestamp(nc):
    ts = repl._get_timestamp()
    dict = {"t": ts, "dt": TICK}
    
    payload = json.dumps(dict).encode()
    await nc.publish("time.tick", payload)
    
async def send_coordinates(state, nc):
    payload = json.dumps(state["coordinates"]).encode()
    await nc.publish("coodrinates", payload)

async def send_velocity(state, nc):
    payload = json.dumps(state["velocity"]).encode()
    await nc.publish("velocity", payload)

async def send_accelaration(state, nc):
    payload = json.dumps(state["acceleration"]).encode()
    await nc.publish("accelaration", payload)
    
async def make_step(nc):
    try:
        repl.do_step(f"{TICK}")
        
        print("do step")

        state_dict = repl.do_get_rocket_state("1")
        
        await send_timestamp(nc)
        await send_coordinates(state_dict, nc)
        await send_velocity(state_dict, nc)
        await send_accelaration(state_dict, nc)
    except Exception as e:
        print(f"[time.tick] Failed to send tick: {e}")

async def ct_state_handler(msg, nc):
    # Print the received engines message
    print(f"[engines] Received message:")
    print(f"  Subject: {msg.subject}")
    print(f"  Data: {msg.data.decode()}")
    print("-" * 40)
    
    engine_params: dict[str, float] = json.loads(msg.subject)
    for engine_name in engine_params.keys():
        repl.do_set_engine_force(f"1 {engine_name} {engine_params[engine_name]}")
    
    # Send tick message after receiving engines
    await make_step(nc)

async def main():
    elapsed_t = 0

    repl.do_init()
    repl.do_animation_init()

    repl.do_add_rocket("id=1 tilt=45 mass=1500") # todo: who sets mass, tilt?
    repl.do_create_vector_transforms("main_vector origin_move=0,-10 r=180")
    repl.do_create_vector_transforms("control_1 origin_move=1,5 r=90")
    repl.do_create_vector_transforms("control_2 origin_move=-1,5 r=270")

    repl.do_create_engine("engine_main main_vector 1500 20000")
    repl.do_create_engine("engine_orientation1 control_1 0 10000")
    repl.do_create_engine("engine_orientation2 control_2 0 10000")

    repl.do_attach_engine("1 engine_main")
    repl.do_attach_engine("1 engine_orientation1")
    repl.do_attach_engine("1 engine_orientation2")
    
    try:
        nc = await nats.connect("nats://localhost:4222")
        print(f"Connected to NATS at nats://nats:4222")
        print(f"Subscribing to 'engines' channel...")

    except Exception as e:
        print(f"Failed to connect to NATS: {e}")
        return

    # Create a closure function to pass nc to the handler
    async def handler(msg):
        await ct_state_handler(msg, nc)

    subscription = await nc.subscribe("engines", cb=handler)
    print("Listening for engines messages. Press Ctrl+C to exit...")
    
    await make_step(nc)
    
    try:
        # Keep the connection alive
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nDisconnecting...")
    finally:
        await nc.drain()

if __name__ == "__main__":
    asyncio.run(main())