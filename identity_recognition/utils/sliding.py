def sliding_window(data, win_size, step_size):
    length = int((data.shape[-1]-win_size)/step_size+1)
    patchs = []
    for i in range(length):
        patch = data[:, i*step_size:i*step_size+win_size]
        patchs.append(patch)
    return patchs