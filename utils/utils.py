from models.basenet import *
import os
import torch
# import neptune
import socket


def get_model_mme(net, num_class=13, temp=0.05, top=False, norm=True):
    dim = 2048
    if "resnet" in net:
        model_g = ResBase(net, top=top)
        if "resnet18" in net:
            dim = 512
        if net == "resnet34":
            dim = 512
    elif "vgg" in net:
        model_g = VGGBase(option=net, pret=True, top=top)
        dim = 4096
    if top:
        dim = 1000
    print("selected network %s"%net)
    return model_g, dim

def log_set(kwargs):
    source_data = kwargs["source_data"]
    target_data = kwargs["target_data"]
    network = kwargs["network"]
    conf_file = kwargs["config_file"]
    script_name = kwargs["script_name"]

    #args = kwargs["args"]

    target_data = os.path.splitext(os.path.basename(target_data))[0]
    logname = "{source}2{target}_{network}".format(source=source_data.split("_")[1],
                                                            target=target_data.split("_")[1],
                                                                               network=network)
    logname = os.path.join("record", kwargs["exp_name"],
                os.path.basename(conf_file).replace(".yaml", "").split("_")[1], logname)
    logname += ".txt"
    if not os.path.exists(os.path.dirname(logname)):
        os.makedirs(os.path.dirname(logname))
    print("record in %s " % logname)
    import logging
    logger = logging.getLogger("GCDDA")
    logging.basicConfig(filename=logname, format="%(message)s")
    logger.setLevel(logging.INFO)
    logger.info("{}_2_{}".format(source_data, target_data))
    return logname, logger


def save_model(model_g, model_c1, model_c2, mlp, save_path):
    save_dic = {
        'g_state_dict': model_g.state_dict(),
        'c1_state_dict': model_c1.state_dict(),
        'c2_state_dict': model_c2.state_dict(),
        'mlp_state_dict': mlp.state_dict(),
    }
    torch.save(save_dic, save_path)

def load_model(model_g, model_c1, model_c2, mlp, load_path):
    import copy
    checkpoint = torch.load(load_path)

    model_g_copy = copy.deepcopy(model_g)
    model_c1_copy = copy.deepcopy(model_c1)
    model_c2_copy = copy.deepcopy(model_c2)
    mlp_copy = copy.deepcopy(mlp)

    model_g_copy.load_state_dict(checkpoint['g_state_dict'])
    model_c1_copy.load_state_dict(checkpoint['c1_state_dict'])
    model_c2_copy.load_state_dict(checkpoint['c2_state_dict'])
    mlp_copy.load_state_dict(checkpoint['mlp_state_dict'])
    return model_g_copy, model_c1_copy, model_c2_copy, mlp_copy