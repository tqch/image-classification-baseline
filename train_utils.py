from models.resnet import *
from models.wideresnet import *
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from copy import deepcopy
from datasets import get_dataloaders


PREDEFINED_MODEL_LIST = {
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
    "wideresnet_28_10": WideResNet_28_10,
    "wideresnet_34_10": WideResNet_34_10,
    "wideresnet_40_10": WideResNet_40_10
}


OPTIMIZER_LIST = {
    "sgd": SGD
}

GENERIC_MODEL_LIST = {
    "resnet": ResNet,
    "wideresnet": WideResNet
}


class DummyScheduler:
    def __init__(self): pass

    def step(self): pass

    def state_dict(self): return None

    def load_state_dict(self, state_dict=None): pass


SCHEDULER_LIST = {
    "lambdalr": LambdaLR,
    "dummy": DummyScheduler
}


def build_model_from_name(model_name):
    return PREDEFINED_MODEL_LIST[model_name]()


def build_model_from_configs(mod_configs):
    cfgs = deepcopy(mod_configs)
    model_type = cfgs.pop("type")
    return GENERIC_MODEL_LIST[model_type](**cfgs)


def build_optimizer(model, opt_configs):
    cfgs = deepcopy(opt_configs)
    return OPTIMIZER_LIST[cfgs.pop("type")](model.parameters(), **cfgs)


def build_scheduler(opt, sch_configs):
    cfgs = deepcopy(sch_configs)
    sch_type = cfgs.pop("type")
    sch_cls = SCHEDULER_LIST[sch_type]
    if sch_type == "lambdalr":
        gamma = cfgs.pop("gamma")
        epochs = cfgs.pop("decay_epochs")

        def lr_lambda(x):
            fct = 0
            for e in epochs:
                if x >= e:
                    fct += 1
            return gamma ** fct
        return sch_cls(opt, lr_lambda=lr_lambda, **cfgs)


def build_dataloader(dat_configs):
    return get_dataloaders(**dat_configs)


def build_all(
        mod_name=None,
        mod_configs=None,
        opt_configs=None,
        sch_configs=None,
        dat_configs=None
):
    if mod_name is not None:
        model = build_model_from_name(mod_name)
    else:
        model = build_model_from_name(mod_configs)
    optimizer = build_optimizer(model, opt_configs)
    scheduler = build_scheduler(optimizer, sch_configs)
    trainloader, testloader = build_dataloader(dat_configs)
    return model, optimizer, scheduler, trainloader, testloader
