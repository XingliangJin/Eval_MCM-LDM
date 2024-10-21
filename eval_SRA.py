import json
import os
from pathlib import Path
import yaml
from src.utils.fixseed import fixseed
import numpy as np
import pytorch_lightning as pl
import torch
from rich import get_console
from rich.table import Table
from omegaconf import OmegaConf
from src.evaluate.tools import save_metrics
from argparse import ArgumentParser
from src.evaluate.stgcn.evaluate import Evaluation as STGCNEvaluation
import src.utils.rotation_conversions as geometry
from tqdm import tqdm

from torch.utils.data import DataLoader
from src.utils.tensors import collate
from src.datasets.get_dataset import get_datasets1, get_datasets1_gt, get_datasets, get_datasets_gen, get_datasets_style

def print_table(title, metrics):
    table = Table(title=title)

    table.add_column("Metrics", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for key, value in metrics.items():
        table.add_row(key, str(value))

    console = get_console()
    console.print(table, justify="center")


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval



def save_dict_to_file(filename, dictionary ):
    with open(filename, 'w') as file:
        for key, value in dictionary.items():
            file.write(f'{key}: {value}\n')



niter = 1



def convert_x_to_rot6d(x, pose_rep):
    # convert rotation to rot6d
    if pose_rep == "rotvec":
        x = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(x))
    elif pose_rep == "rotmat":
        x = x.reshape(*x.shape[:-1], 3, 3)
        x = geometry.matrix_to_rotation_6d(x)
    elif pose_rep == "rotquat":
        x = geometry.matrix_to_rotation_6d(geometry.quaternion_to_matrix(x))
    elif pose_rep == "rot6d":
        x = x
    else:
        raise NotImplementedError("No geometry for this one.")
    return x


class NewDataloader:
    def __init__(self, mode, model, parameters, dataiterator, device):
        # assert mode in ["gen", "rc", "gt"]

        pose_rep = parameters["pose_rep"]
        translation = parameters["translation"]

        self.batches = []

        with torch.no_grad():
            for databatch in tqdm(dataiterator, desc=f"Construct dataloader: {mode}.."):
                if mode == "trans" or "ori":
                    # classes = databatch["y"]
                    # gendurations = databatch["lengths"]
                    # batch = model.generate(classes, gendurations)
                    # feats = "output"
                    batch = {key: val.to(device) for key, val in databatch.items()}
                    feats = "x"
                elif mode == "gt":
                    batch = {key: val.to(device) for key, val in databatch.items()}
                    feats = "x"
                elif mode == "rc":
                    databatch = {key: val.to(device) for key, val in databatch.items()}
                    batch = model(databatch)
                    feats = "output"

                batch = {key: val.to(device) for key, val in batch.items()}

                if translation:
                    x = batch[feats][:, :-1]
                else:
                    x = batch[feats]

                x = x.permute(0, 3, 1, 2)
                # x = convert_x_to_rot6d(x, pose_rep)
                x = x.permute(0, 2, 3, 1)

                batch["x"] = x

                self.batches.append(batch)

    def __iter__(self):
        return iter(self.batches)










def load_args(filename):
    with open(filename, "rb") as optfile:
        opt = yaml.load(optfile, Loader=yaml.Loader)
    return opt


def main():

    # args
    parser = ArgumentParser()

    group = parser.add_argument_group("Params")
    group.add_argument(
        "--pkl_path",
        type=str,
        default="eval_source/style_epoch_399_scale_2-5.pkl",
        required=False,
        help="ply set",
    )
    params = parser.parse_args()
    pkl_path = params.pkl_path






    seed = 10
    fixseed(seed)
# for gt
    gtparameters = load_args("save_style_recog/opt.yaml")
    gtparameters["device"]=torch.device("cuda:0")
    dataset_name = gtparameters["dataset"]


# load STGCN
    stgcnevaluation = STGCNEvaluation(dataset_name, gtparameters, device=gtparameters["device"],  modelpath="save_style_recog/checkpoint_0050.pth.tar")
    datasetGT = get_datasets(gtparameters)["train"]
    dataiterator_gt = DataLoader(datasetGT, batch_size=128,
                                    shuffle=False, num_workers=8,
                                    collate_fn=collate)
    

# for gen
    recogparameters = load_args("save_style_recog/opt.yaml")
    recogparameters["device"]=torch.device("cuda:0")
    dataset_name = recogparameters["dataset"]
    recogparameters["eval_motion_path"] = pkl_path # for ours


    # for gen
    # for trans sra
    recogparameters_trans = recogparameters
    recogparameters_trans['eval_type'] = 'trans'
    datasettrans = get_datasets_style(recogparameters)["train"]

    
    dataiterator_trans = DataLoader(datasettrans, batch_size=128,
                                    shuffle=False, num_workers=8,
                                    collate_fn=collate)
    
    # for ori sra
    recogparameters_ori = recogparameters
    recogparameters_ori['eval_type'] = 'ori'
    datasetori = get_datasets_style(recogparameters)["train"]

    
    dataiterator_ori = DataLoader(datasetori, batch_size=128,
                                    shuffle=False, num_workers=8,
                                    collate_fn=collate)






    model = None
    gtLoaders = NewDataloader("gt", model, gtparameters,
                                        dataiterator_gt,
                                        recogparameters["device"])
    
    transLoaders = NewDataloader("trans", model, recogparameters_trans,
                                        dataiterator_trans,
                                        recogparameters_trans["device"])
    oriLoaders = NewDataloader("ori", model, recogparameters_ori,
                                        dataiterator_ori,
                                        recogparameters_ori["device"])



    allseeds = list(range(niter))

    loaders = {"gt": gtLoaders,
               "trans": transLoaders,
               "ori": oriLoaders}
    flag = 's'
    stgcn_metrics, _ = stgcnevaluation.evaluate(model, loaders, flag)

    print("***********")
    print(recogparameters["eval_motion_path"].split('/')[-1].split('.')[0])
    print("***********")
    print(stgcn_metrics)


    save_dict_to_file("eval_results/sra_{}.txt".format(recogparameters["eval_motion_path"].split('/')[-1].split('.')[0]), stgcn_metrics)





if __name__ == "__main__":
    main()
