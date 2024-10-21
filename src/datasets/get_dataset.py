def get_dataset(name="ntu13"):
    if name == "ntu13":
        from .ntu13 import NTU13
        return NTU13
    elif name == "uestc":
        from .uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml3d_cmu":
        from .humanact12poses import humanml3d_cmu
        return humanml3d_cmu
    elif name == "humanml3d_cmu_style":
        from .humanact12poses import humanml3d_cmu_style
        return humanml3d_cmu_style
    elif name == "humanml3d_eval":
        from .humanact12poses import humanml3d_eval
        return humanml3d_eval
    elif name == "humanml3d_gt":
        from .humanact12poses import humanml3d_gt
        return humanml3d_gt
    elif name == "humanml3d_gen":
        from .humanact12poses import humanml3d_gen
        return humanml3d_gen
    elif name == "humanml3d_style":
        from .humanact12poses import humanml3d_style
        return humanml3d_style


def get_datasets(parameters):
    name = parameters["dataset"]

    DATA = get_dataset(name)
    dataset = DATA(split="train", **parameters)

    train = dataset

    # test: shallow copy (share the memory) but set the other indices
    from copy import copy
    test = copy(train)
    test.split = test

    datasets = {"train": train,
                "test": test}

    # add specific parameters from the dataset loading
    dataset.update_parameters(parameters)

    return datasets


def get_datasets1(parameters):
    name = "humanml3d_eval"
    eval_motion_path = parameters["eval_motion_path"]

    DATA = get_dataset(name)
    dataset = DATA(eval_motion_path,split="train", **parameters)

    train = dataset

    # test: shallow copy (share the memory) but set the other indices
    from copy import copy
    test = copy(train)
    test.split = test

    datasets = {"train": train,
                "test": test}

    # add specific parameters from the dataset loading
    dataset.update_parameters(parameters)

    return datasets

def get_datasets1_gt(parameters):
    name = "humanml3d_gt"
    eval_motion_path = parameters["eval_motion_path"]

    DATA = get_dataset(name)
    dataset = DATA(eval_motion_path,split="train", **parameters)

    train = dataset

    # test: shallow copy (share the memory) but set the other indices
    from copy import copy
    test = copy(train)
    test.split = test

    datasets = {"train": train,
                "test": test}

    # add specific parameters from the dataset loading
    dataset.update_parameters(parameters)

    return datasets



def get_datasets_gen(parameters):
    name = "humanml3d_gen"
    eval_motion_path = parameters["eval_motion_path"]

    DATA = get_dataset(name)
    dataset = DATA(eval_motion_path,split="train", **parameters)

    train = dataset

    # test: shallow copy (share the memory) but set the other indices
    from copy import copy
    test = copy(train)
    test.split = test

    datasets = {"train": train,
                "test": test}

    # add specific parameters from the dataset loading
    dataset.update_parameters(parameters)

    return datasets

def get_datasets_style(parameters):
    name = "humanml3d_style"
    eval_motion_path = parameters["eval_motion_path"]
    eval_type = parameters["eval_type"]

    DATA = get_dataset(name)
    dataset = DATA(eval_motion_path,eval_type,split="train", **parameters)

    train = dataset

    # test: shallow copy (share the memory) but set the other indices
    from copy import copy
    test = copy(train)
    test.split = test

    datasets = {"train": train,
                "test": test}

    # add specific parameters from the dataset loading
    dataset.update_parameters(parameters)

    return datasets