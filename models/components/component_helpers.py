# circular import errors forced us to create this separate helper file 
# even though it would of course be more elegant to have all helpers in the main helper file (helpers.py)

import torch

# loads weighs sitting at path into an empty nn.Module (model)
def load_weights(path, model, device):
    if device == torch.device('cpu'): model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    else: model.load_state_dict(torch.load(path))
    return model

# check if exit command has been entered into the file "flag.txt"
# 
# this could probably be implemented more elegantly by launching a child process that scans the cmd line (then we could enter the stop cmd into the terminal)
# but I am not willing to get into multiprocessing bugs just for a slightly more elegant way of stopping training. 
def check_exit():
    with open("/cluster/home/elucas/thesis/flag.txt", 'r+') as f:
        content = f.read().strip()
        if content.lower() == "e":
            # remove file contents s.t. next training section does not immediately also exit
            f.truncate(0)
            f.seek(0)
            return True
    return False

# turns a label (2 or 3 possible categories) into a 2 or 3 dimensional 1-hot vector
def to_distrib(batch_size, labels, binary_mode):
    zeros = torch.zeros((batch_size, 2)) if binary_mode else torch.zeros((batch_size, 3))
    for i, label in enumerate(labels):
        zeros[i][int(label.item())] = 1  # int cast should always be fine as label should always be int (packaged into a float for some reason)
    return zeros