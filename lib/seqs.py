#################################################################
def chunker(seq, chunk_size):
    chunk = list()
    size = 0
    for item in seq:
        chunk.append(item)
        size += 1
        if size % chunk_size == 0:
            yield chunk
            chunk = list()
            size = 0
    if size > 0:
        yield chunk

#################################################################
def get_bigrams(seq):
    for i in range(len(seq)-1):
        yield (seq[i], seq[i+1])