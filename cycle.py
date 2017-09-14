from itertools import cycle
import numpy as np

def foo():


videos = [np.random.randint(10,99,(108,)),
          np.random.randint(10,99,(31,)),
          np.random.randint(10,99, (65,)),
          np.random.randint(10,99, (100,)),
          np.random.randint(10,99, (21,))]

frameskip = 5.0
N = sum([len(vid) for vid in videos])
total_nr_frames = N//frameskip

batch_size = 8
nb_batches = int(total_nr_frames) // batch_size
X = np.zeros((nb_batches, (total_nr_frames // nb_batches)), dtype=np.int64)
print total_nr_frames, np.prod(X.shape)

perm = np.random.permutation(nb_batches)
pool = cycle(perm)

cnts = np.zeros((nb_batches,), dtype=np.int32)
for vid in videos:
    # get one every other "frameskip" frames
    x = vid[::int(frameskip)]
    # determine batch assignment: batch indices for every frame
    batch_inds = np.array([pool.next() for i in range(len(x))])

    # ---
    # Chunk block assignment
    # ---
    nb_chunks = int(np.ceil(len(batch_inds) / float(nb_batches)))
    ptr = 0
    for k in range(nb_chunks):
        batch_inds_k = batch_inds[ptr:ptr + nb_batches]
        x_chunk = x[ptr:ptr + nb_batches]
        cnts_k = cnts[batch_inds_k]
        try:
            X[batch_inds_k, cnts_k] = x_chunk
        except IndexError:
            mask = (cnts_k == min(cnts_k))
            X[batch_inds_k[mask], cnts_k[mask]] = x_chunk[mask]
            pass
        cnts[batch_inds_k] += 1

        ptr += nb_batches
    # ---
    # alt: one-by-one assignment
    # ---
    # for i,ind in enumerate(inds):
    #     try:
    #         X[ind, ptrs[ind]] = x[i]
    #         ptrs[ind] += 1
    #     except IndexError, e:
    #         pass
    # ---

print X
print cnts