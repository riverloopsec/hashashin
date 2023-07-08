import subprocess
import argparse

demo_steps = [
    "hashashin -db",
    "hashashin --summary",
]


def safedocs_demo():
    # take cli arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("step", help="demo step number to run", type=int)
    parser.add_argument(
        "-x", "--execute", help="execute the demo step", action="store_true"
    )
    args = parser.parse_args()

    shell_fmt_str = "$ %s"

    step = demo_steps[args.step]
    print(shell_fmt_str % step)
    if args.execute:
        subprocess.run(["bash", "-c", step])
