import subprocess
import os
import time

def start_tensorboard(logdir='./runs', port=6006):
    if os.name == 'nt':  # Check if the operating system is Windows
        # Windows specific command to start TensorBoard
        command = f'tensorboard --logdir={logdir} --port={port}'
        proc = subprocess.Popen(command, shell=True)
    else:
        # UNIX-like operating system command
        proc = subprocess.Popen(['tensorboard', '--logdir', logdir, '--port', str(port)])
    time.sleep(5)  # Wait for TensorBoard to start
    return f"http://localhost:{port}/"

tensorboard_url = start_tensorboard()
print(f"TensorBoard URL: {tensorboard_url}")
