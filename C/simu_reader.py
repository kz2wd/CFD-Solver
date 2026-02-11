import mmap
import struct

import numpy as np
import posix_ipc

X = 64
N = X * X * 2

steps = 200
sample_interval = 1

record_size = steps // sample_interval
output_array = np.zeros((record_size, X, X, 2))

shm = posix_ipc.SharedMemory("/sim_shm", posix_ipc.O_CREAT, size=8 + 8 * N)
mm = mmap.mmap(shm.fd, shm.size)

shm.close_fd()

mm.write(struct.pack("Q", 0))

write_idx = np.frombuffer(mm, dtype=np.uint64, count=1, offset=0)
data = np.frombuffer(mm, dtype=np.float64, count=N, offset=8)

last = 0
try:
    while True:
        w = int(write_idx[0])
        index = w // sample_interval
        if w > last:
            last = w
            frame = data.copy().reshape(X, X, 2)
            print(frame.shape)
            print(f"reading step {w}, saving index {index}")
            print(f"stats: {frame.mean()}")

            output_array[index] = frame
        if index >= record_size - 1:
            break
except KeyboardInterrupt:
    pass
print("Saving run file")
np.save("runs/out", output_array)
