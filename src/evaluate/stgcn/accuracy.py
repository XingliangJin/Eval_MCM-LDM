import torch


# def calculate_accuracy(model, motion_loader, num_labels, classifier, device):
#     confusion = torch.zeros(num_labels, num_labels, dtype=torch.long)
#     with torch.no_grad():
#         for batch in motion_loader:
#             batch_prob = classifier(batch)["yhat"]
#             batch_pred = batch_prob.max(dim=1).indices
#             for label, pred in zip(batch["y"], batch_pred):
#                 confusion[label][pred] += 1

#     accuracy = torch.trace(confusion)/torch.sum(confusion)
#     return accuracy.item(), confusion.cpu().numpy().tolist()



def calculate_accuracy(model, motion_loader, num_labels, classifier, device):
    confusion = torch.zeros(num_labels, num_labels, dtype=torch.long)
    correct_top1 = 0
    correct_top2 = 0
    total = 0

    with torch.no_grad():
        for batch in motion_loader:
            batch_prob = classifier(batch)["yhat"]
            _, batch_pred_top2 = torch.topk(batch_prob, k=2, dim=1)
            batch_pred_top1 = batch_pred_top2[:, 0]
            batch_pred_top2 = batch_pred_top2[:, 1]

            for label, pred_top1, pred_top2 in zip(batch["y"], batch_pred_top1, batch_pred_top2):
                confusion[label][pred_top1] += 1
                # confusion[label][pred_top2] += 1

                if label == pred_top1:
                    correct_top1 += 1
                if label == pred_top1 or label == pred_top2:
                    correct_top2 += 1

                total += 1

    accuracy_top1 = correct_top1 / total
    accuracy_top2 = correct_top2 / total

    return accuracy_top1, accuracy_top2, confusion.cpu().numpy().tolist()