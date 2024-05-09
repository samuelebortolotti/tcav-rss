import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from models import MiniBOIACnn
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

from datasets.miniboia import MINIBOIA

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
                inputs, labels, concepts = inputs.to(device), labels, concepts.to(device)
                new_labels = []
                for i in range(labels.size(0)):
                    new_labels.append(torch.tensor(int(''.join(map(str, labels[i].tolist())), 2)))
                labels = torch.stack(new_labels, dim=0)
                labels = labels.to(device)

                outputs = model(inputs)
                outputs = torch.softmax(outputs, dim=1)
                predicted = outputs.max(dim=1)[1]
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
            inputs, labels, concepts = inputs.to(device), labels, concepts.to(device)
            new_labels = []
            for i in range(labels.size(0)):
                new_labels.append(torch.tensor(int(''.join(map(str, labels[i].tolist())), 2)))
            labels = torch.stack(new_labels, dim=0)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # save model parameters
    torch.save(best_weights, 'best.pth')


def validate(model, full_images):
    model.eval()
    weights = torch.load('best.pth')
    model.load_state_dict(weights)
    extract_layer = ['conv4', 'fc1', 'fc2', 'fc3']
    model = ModelWrapper(model, extract_layer)

    scorer = TCAV(model, validloader, concept_dict, class_dict.values(), 150)

    print('Generating concepts...')
    scorer.generate_activations(extract_layer)
    scorer.load_activations()
    print('Concepts successfully generated and loaded!')

    suffix = ''
    if full_images:
        suffix = '_full_images'

    print('Calculating TCAV scores...')
    for layer in extract_layer:
        print("Considering layer", layer)
        scorer.generate_cavs(layer)
        scorer.calculate_tcav_score(layer, f'output/tcav_result_boia{suffix}_{layer}.npy')
        loaded_scores = np.load(f'output/tcav_result_boia{suffix}_{layer}.npy')
        scores = loaded_scores.T.tolist()

        table = pt.PrettyTable()
        table.field_names = ['class'] + list(concept_dict.keys())
        for i, k in enumerate(class_dict.keys()):
            new_row = [k] + scores[i]
            table.add_row(new_row)
        print(table)
    print('Done!')


if __name__ == "__main__":
    full_images = True

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    from argparse import Namespace
    args = Namespace(
        preprocess=0,
        finetuning=0,
        batch_size=256,
        n_epochs=20,
        validate=1,
        dataset="miniboia",
        lr=0.1,
        exp_decay=1,
        warmup_steps=1,
        wandb=None,
        task="boia",
        backbone="conceptizer",
        c_sup=1,
        which_c=[-1],
    )

    # import the miniboia dataset
    dataset = MINIBOIA(args)
    trainloader, _, testloader = dataset.get_data_loaders()

    class_dict = {
        'forward': int(''.join(map(str,[1, 0, 0, 0])), 2),
        'stop': int(''.join(map(str,[0, 1, 0, 0])), 2),
        'f_left_right': int(''.join(map(str,[1, 0, 1, 1])), 2),
        's_left_right': int(''.join(map(str,[0, 1, 1, 1])), 2),
        's_left': int(''.join(map(str,[0, 1, 1, 0])), 2),
        'f_left': int(''.join(map(str,[1, 0, 1, 0])), 2),
        's_right': int(''.join(map(str,[0, 1, 0, 1])), 2),
        'f_right': int(''.join(map(str,[1, 0, 0, 1])), 2),
    }

    reverse_class_dict = {v : k for k, v in class_dict.items()}

    validate_dataset = ValidateDataset(class_dict.values(), trainloader, "boia")
    validloader = DataLoader(validate_dataset, batch_size=1, shuffle=False, num_workers=1)

    concept_dict = {}

    folder_suffix = ''

    if full_images:
        folder_suffix = '-preprocess-full'

    for dirname in os.listdir(f'../data/miniboia{folder_suffix}/concepts'):
        fullpath = os.path.join(f'../data/miniboia{folder_suffix}/concepts', dirname)
        if os.path.isdir(fullpath):
            concept_dict[dirname] = data_loader(fullpath)

    model = MiniBOIACnn()
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()

    train()
    validate(model, full_images)