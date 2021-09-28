from collections import OrderedDict


def load_state_dict_with_replace(state_dict, own_state):
    new_state = OrderedDict()
    for name, param in state_dict.items():
        if name in own_state:
            new_state[name] = param
    return new_state
