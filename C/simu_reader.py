import configparser
import mmap
import struct
import time
from dataclasses import asdict, dataclass

import numpy as np
import posix_ipc


@dataclass
class SimulationParameters:
    re: float
    N: int
    steps: int
    sampling: int
    K: float
    dt: float
    scheme: str


params = SimulationParameters(100.0, 129, 50000, 1, 1.0, 1e-5, "IMEX")

config = configparser.ConfigParser()
config["Simulation"] = asdict(params)
with open("simulation.ini", "w") as configfile:
    config.write(configfile)

record_size = params.steps // params.sampling
output_array = np.zeros((record_size, params.N, params.N, 2))

shm = posix_ipc.SharedMemory(
    "/sim_shm", posix_ipc.O_CREAT, size=8 + 8 * params.N * params.N * 2
)
mm = mmap.mmap(shm.fd, shm.size)

shm.close_fd()

mm.write(struct.pack("Q", 0))

write_idx = np.frombuffer(mm, dtype=np.uint64, count=1, offset=0)
data = np.frombuffer(mm, dtype=np.float64, count=params.N * params.N * 2, offset=8)

last = 0
try:
    while True:
        w = int(write_idx[0])
        index = w // params.sampling
        if w > last:
            last = w
            frame = data.copy().reshape(params.N, params.N, 2)
            print(f"reading step {w}, saving index {index}")
            print(f"stats: {frame.mean()}")

            output_array[index] = frame
        if index >= record_size - 1:
            break
        time.sleep(0.001)
except KeyboardInterrupt:
    pass
print("Saving run file")
np.save("runs/out", output_array)
