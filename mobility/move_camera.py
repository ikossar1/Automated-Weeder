"""
Drive the machine by hand, to point the camera before a photo.

This is the jog script from the Pi, kept here because framing a shot
is a camera job: you move the head until the bed fills the frame, take
the picture, and only then is there anything to map.

    python3 move_camera.py

    arrow up/down     motor 1       arrow left/right  motor 2
    j / k             stepper       a                 solenoid
    q                 quit

Pins are the ones the machine is actually wired to - the same numbers
mobility/run_plan.py drives, confirmed by running each motor and
watching which axis moved:

    23, 24   pan        27, 22   tilt
    25, 6    rail stepper (step, dir)
    26, 5    valve, a latching one: a pulse opens it, a pulse the
             other way shuts it
"""
import time

import curses
import RPi.GPIO as GPIO

IN1, IN2 = 23, 24            # pan
IN3, IN4 = 27, 22            # tilt
SOL_IN1, SOL_IN2 = 26, 5     # valve, through an H-bridge channel
STEP_PIN, DIR_PIN = 25, 6    # rail, on an EasyDriver

MOTOR_PULSE = 0.25           # seconds a DC motor runs per press
SOL_PULSE = 1.0              # seconds to flip the latching valve
STEPS_PER_PRESS = 600        # stepper steps per press
STEP_DELAY = 0.002           # seconds per half pulse, bigger = slower

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
OUTPUTS = [IN1, IN2, IN3, IN4, SOL_IN1, SOL_IN2, STEP_PIN, DIR_PIN]
for pin in OUTPUTS:
    GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)


def stop_all():
    for pin in OUTPUTS:
        GPIO.output(pin, GPIO.LOW)


def pulse(a, b, forward, seconds):
    GPIO.output(a, GPIO.LOW)
    GPIO.output(b, GPIO.LOW)
    GPIO.output(a if forward else b, GPIO.HIGH)
    time.sleep(seconds)
    GPIO.output(a, GPIO.LOW)
    GPIO.output(b, GPIO.LOW)


def step(n, forward):
    GPIO.output(DIR_PIN, GPIO.HIGH if forward else GPIO.LOW)
    time.sleep(0.001)
    for _ in range(n):
        GPIO.output(STEP_PIN, GPIO.HIGH)
        time.sleep(STEP_DELAY)
        GPIO.output(STEP_PIN, GPIO.LOW)
        time.sleep(STEP_DELAY)


def main(stdscr):
    stdscr.nodelay(True)
    stdscr.keypad(True)
    stdscr.addstr(0, 0, "Up/Down: pan | Left/Right: tilt | J/K: rail | "
                        "A: valve | q: quit")

    while True:
        key = stdscr.getch()
        if key == -1:
            time.sleep(0.01)
            continue
        if key == ord("q"):
            break

        msg = None
        if key == curses.KEY_UP:
            msg = "pan forward"
            pulse(IN1, IN2, True, MOTOR_PULSE)
        elif key == curses.KEY_DOWN:
            msg = "pan back"
            pulse(IN1, IN2, False, MOTOR_PULSE)
        elif key == curses.KEY_LEFT:
            msg = "tilt forward"
            pulse(IN3, IN4, True, MOTOR_PULSE)
        elif key == curses.KEY_RIGHT:
            msg = "tilt back"
            pulse(IN3, IN4, False, MOTOR_PULSE)
        elif key in (ord("j"), ord("J")):
            msg = "rail forward"
            step(STEPS_PER_PRESS, True)
        elif key in (ord("k"), ord("K")):
            msg = "rail back"
            step(STEPS_PER_PRESS, False)
        elif key in (ord("a"), ord("A")):
            msg = "valve pulsed"
            pulse(SOL_IN1, SOL_IN2, True, SOL_PULSE)

        if msg:
            curses.flushinp()        # drop key repeats queued while it ran
            stdscr.move(2, 0)
            stdscr.clrtoeol()
            stdscr.addstr(2, 0, msg)
            stdscr.refresh()


try:
    curses.wrapper(main)
finally:
    # The valve latches, so a pulse the other way is the only thing that
    # shuts it - dropping the pins would leave it open.
    pulse(SOL_IN2, SOL_IN1, True, SOL_PULSE)
    stop_all()
    GPIO.cleanup()
    print("Motors stopped, valve shut.")
