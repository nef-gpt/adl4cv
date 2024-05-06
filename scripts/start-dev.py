#!/usr/bin/env python
import os
from tensorboard import program


def start_tensorboard():
    """
    Script called from poetry to run tensorboard
    """
    tb = program.TensorBoard()
    # use cwd/runs as default logdir
    tb.configure(argv=[None, "--logdir", "./runs"])
    url = tb.launch()
    print(f"TensorBoard is available at {url}")

    # wait for the user to terminate the process
    try:
        input("Press Enter to stop TensorBoard...")
    except KeyboardInterrupt:
        pass
    finally:
        tb._shutdown_server()
        print("TensorBoard has been stopped.")


if __name__ == "__main__":
    start_tensorboard()
