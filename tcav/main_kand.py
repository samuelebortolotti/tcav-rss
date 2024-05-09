import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from models import KANDCNN
from tcav import TCAV
from model_wrapper import ModelWrapper
from mydata import MyDataset, ValidateDataset
import os
import sys
import prettytable as pt
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.append("/mnt/cimec-storage6/users/samuele.bortolotti/RSs/miniBoiaRss/mini-boia-performances/rss")
sys.path.append("/mnt/cimec-storage6/users/samuele.bortolotti/RSs/miniBoiaRss/mini-boia-performances/rss/datasets")

from datasets.minikandinsky import MiniKandinsky

def data_loader(base_path):
    image_dataset_train = MyDataset(base_path, transform=data_transforms)
    train_loader = DataLoader(image_dataset_train, batch_size=1)
    return train_loader


def train():
    best_weights = model.state_dict()
    best_acc = 0.0
    for epoch in range(10):
        # test phase
        total = 0
        score = 0
        f1 = 0
        with torch.no_grad():
            model.eval()
            for inputs, labels, concepts in testloader:
                inputs, labels, concepts = inputs.to(device), labels.to(device), concepts.to(device)
                
                outputs = model(inputs)
                outputs = torch.sigmoid(outputs)
                predicted = (outputs > 0).float()
                predicted = predicted.squeeze(-1)

                # labels for kandinsky
                labels = labels[:, -1]

                total += labels.size(0)
                score += predicted.eq(labels).sum().item()

        acc = score / total
        print("epoch: {}\tacc: {}".format(epoch, acc))

        if acc > best_acc:
            best_acc = acc
            best_weights = model.state_dict()

        # train phase
        model.train()

        for inputs, labels, concepts in trainloader:
            inputs, labels, concepts = inputs.to(device), labels.to(device), concepts.to(device)
            
            # per kand
            labels = labels[:, -1].float()

            optimizer.zero_grad()
            outputs = torch.sigmoid(model(inputs)).squeeze(-1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # save model parameters
    torch.save(best_weights, 'best.pth')


def validate(model):
    model.eval()
    weights = torch.load('best.pth')
    model.load_state_dict(weights)
    extract_layer = 'fc1'
    model = ModelWrapper(model, [extract_layer])

    scorer = TCAV(model, validloader, concept_dict, class_dict.values(), 150)

    print('Generating concepts...')
    scorer.generate_activations([extract_layer])
    scorer.load_activations()
    print('Concepts successfully generated and loaded!')

    print('Calculating TCAV scores...')
    scorer.generate_cavs(extract_layer)
    scorer.calculate_tcav_score(extract_layer, 'output/tcav_result_kand.npy')
    loaded_scores = np.load('output/tcav_result_kand.npy')
    scores = loaded_scores.T.tolist()
    print('Done!')

    table = pt.PrettyTable()
    table.field_names = ['class'] + list(concept_dict.keys())
    for i, k in enumerate(class_dict.keys()):
        new_row = [k] + scores[i]
        table.add_row(new_row)
    print(table)


if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28, 252))
    ])

    from argparse import Namespace
    args = Namespace(
        preprocess=0,
        finetuning=0,
        batch_size=256,
        n_epochs=20,
        validate=1,
        dataset="minikandinksy",
        lr=0.001,
        exp_decay=1,
        warmup_steps=1,
        wandb=None,
        task="kand",
        backbone="conceptizer",
        c_sup=1,
        which_c=[-1],
    )

    # import the miniboia dataset
    dataset = MiniKandinsky(args)
    trainloader, _, testloader = dataset.get_data_loaders()

    class_dict = {
        'true': 1,
        'false': 0
    }

    reverse_class_dict = {v : k for k, v in class_dict.items()}

    validate_dataset = ValidateDataset(class_dict.values(), trainloader, "kand")
    validloader = DataLoader(validate_dataset, batch_size=1, shuffle=False, num_workers=1)

    concept_dict = {}
    for dirname in os.listdir('../data/kand-preprocess/concepts'):
        fullpath = os.path.join('../data/kand-preprocess/concepts', dirname)
        if os.path.isdir(fullpath):
            concept_dict[dirname] = data_loader(fullpath)

    model = KANDCNN()
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()

    train()
    validate(model)