import torch
import numpy as np
from .accuracy import calculate_accuracy
from .fid import calculate_fid
from .diversity import calculate_diversity_multimodality

from src.recognition.models.stgcn import STGCN


class Evaluation:
    def __init__(self, dataname, parameters, device,modelpath, seed=None):
        layout = "smpl" if parameters["glob"] else "smpl_noglobal"
        model = STGCN(in_channels=parameters["nfeats"],
                      num_class=parameters["num_classes"],
                      graph_args={"layout": layout, "strategy": "spatial"},
                      edge_importance_weighting=True,
                      device=parameters["device"])

        model = model.to(parameters["device"])


        state_dict = torch.load(modelpath, map_location=parameters["device"])
        model.load_state_dict(state_dict)
        model.eval()

        self.num_classes = parameters["num_classes"]
        self.model = model

        self.dataname = dataname
        self.device = device

        self.seed = seed

    def compute_features(self, model, motionloader):
        # calculate_activations_labels function from action2motion
        activations = []
        labels = []
        with torch.no_grad():
            for idx, batch in enumerate(motionloader):
                activations.append(self.model(batch)["features"])
                labels.append(batch["y"])
            activations = torch.cat(activations, dim=0)
            labels = torch.cat(labels, dim=0)
        return activations, labels

    @staticmethod
    def calculate_activation_statistics(activations):
        activations = activations.cpu().numpy()
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    def evaluate(self, model, loaders, flag):
        def print_logs(metric, key):
            print(f"Computing stgcn {metric} on the {key} loader ...")

        metrics_all = {}
        for sets in ["train"]:
            computedfeats = {}
            metrics = {}
            for key, loaderSets in loaders.items():
                loader = loaderSets

                metric = "accuracy"
                print_logs(metric, key)
                mkey1 = f"{metric}_{key}_top1"
                mkey2 = f"{metric}_{key}_top2"
                con = "confusion"
                con_key = f"{con}_{key}"
                metrics[mkey1], metrics[mkey2], _ = calculate_accuracy(model, loader,
                                                      self.num_classes,
                                                      self.model, self.device)
                
                # features for diversity
                print_logs("features", key)
                feats, labels = self.compute_features(model, loader)
                print_logs("stats", key)
                stats = self.calculate_activation_statistics(feats)

                computedfeats[key] = {"feats": feats,
                                      "labels": labels,
                                      "stats": stats}

                # print_logs("diversity", key)
                # ret = calculate_diversity_multimodality(feats, labels, self.num_classes,
                #                                         seed=self.seed)
                # metrics[f"diversity_{key}"], metrics[f"multimodality_{key}"] = ret

            # taking the stats of the ground truth and remove it from the computed feats
            gtstats = computedfeats["gt"]["stats"]
            # computing fid
            for key, loader in computedfeats.items():
                metric = "fid"
                mkey = f"{metric}_{key}"

                stats = computedfeats[key]["stats"]
                if flag == 'c':
                # metrics[mkey] = float(0.01)
                    metrics[mkey] = float(calculate_fid(gtstats, stats))
                else:
                    # don't cal fid when cal sra
                    metrics[mkey] = float(0.01)


            metrics_all[sets] = metrics

        # metrics = {}
        # for sets in ["train", "test"]:
        #     for key in metrics_all[sets]:
        #         metrics[f"{key}_{sets}"] = metrics_all[sets][key]
        return metrics, computedfeats
