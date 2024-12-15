import os
import argparse
import random
import torch
import torch.utils.data
import torch.nn as nn

from tqdm import tqdm
from torchsummary import summary
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
from utils import datasets
from utils.arff_reader import initialize_dataset, initialize_other_dataset

from models.HMCN_F import HMCNFModel


def main():

    parser = argparse.ArgumentParser(description="Train neural network wutg train and validation set, and test it on the test set")

    # Required  parameter
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        required=True,
        choices=[
            "cellcycle_FUN",
            # "church_FUN",
            "derisi_FUN",
            "eisen_FUN",
            # "expr_FUN",
            # "gasch1_FUN",
            "gasch2_FUN",
            # "pheno_FUN",
            # "seq_FUN",
            # "spo_FUN",
            # "cellcycle_GO",
            # "derisi_GO",
            # "eisen_GO",
            # "expr_GO",
            # "gasch1_GO",
            # "gasch2_GO",
            # "pheno_GO",
            # "seq_GO",
            # "spo_GO",
        ],
        help='dataset name, must end with: "_GO", "_FUN", or "_others"',
    )
    # Other parameters
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument("--device", type=str, default="0", help="GPU (default:0)")
    args = parser.parse_args()

    assert "_" in args.dataset
    assert "FUN" in args.dataset or "GO" in args.dataset or "others" in args.dataset

    # Load train, val and test set
    dataset_name = args.dataset
    data = dataset_name.split("_")[0]
    ontology = dataset_name.split("_")[1]

    # Dictionaries with number of features and number of labels for each dataset
    input_dims = {
        "diatoms": 371,
        "enron": 1001,
        "imclef07a": 80,
        "imclef07d": 80,
        "cellcycle": 77,
        "derisi": 63,
        "eisen": 79,
        "expr": 561,
        "gasch1": 173,
        "gasch2": 52,
        "seq": 529,
        "spo": 86,
    }
    output_dims_FUN = {
        "cellcycle": 499,
        "derisi": 499,
        "eisen": 461,
        "expr": 499,
        "gasch1": 499,
        "gasch2": 499,
        "seq": 499,
        "spo": 499,
    }
    output_dims_GO = {
        "cellcycle": 4122,
        "derisi": 4116,
        "eisen": 3570,
        "expr": 4128,
        "gasch1": 4122,
        "gasch2": 4128,
        "seq": 4130,
        "spo": 4116,
    }
    output_dims_others = {"diatoms": 398, "enron": 56, "imclef07a": 96, "imclef07d": 46, "reuters": 102}
    output_dims = {"FUN": output_dims_FUN, "GO": output_dims_GO, "others": output_dims_others}

    # Dictionaries with the hyperparameters associated to each dataset
    hidden_dims_FUN = {
        "cellcycle": 500,
        "derisi": 500,
        "eisen": 500,
        "expr": 1250,
        "gasch1": 1000,
        "gasch2": 500,
        "seq": 2000,
        "spo": 250,
    }
    hidden_dims_GO = {
        "cellcycle": 1000,
        "derisi": 500,
        "eisen": 500,
        "expr": 4000,
        "gasch1": 500,
        "gasch2": 500,
        "seq": 9000,
        "spo": 500,
    }
    hidden_dims_others = {"diatoms": 2000, "enron": 1000, "imclef07a": 1000, "imclef07d": 1000}
    hidden_dims = {"FUN": hidden_dims_FUN, "GO": hidden_dims_GO, "others": hidden_dims_others}
    lrs_FUN = {
        "cellcycle": 1e-4,
        "derisi": 1e-4,
        "eisen": 1e-4,
        "expr": 1e-4,
        "gasch1": 1e-4,
        "gasch2": 1e-4,
        "seq": 1e-4,
        "spo": 1e-4,
    }
    lrs_GO = {
        "cellcycle": 1e-4,
        "derisi": 1e-4,
        "eisen": 1e-4,
        "expr": 1e-4,
        "gasch1": 1e-4,
        "gasch2": 1e-4,
        "seq": 1e-4,
        "spo": 1e-4,
    }
    lrs_others = {"diatoms": 1e-5, "enron": 1e-5, "imclef07a": 1e-5, "imclef07d": 1e-5}
    lrs = {"FUN": lrs_FUN, "GO": lrs_GO, "others": lrs_others}
    epochss_FUN = {
        "cellcycle": 106,
        "derisi": 67,
        "eisen": 110,
        "expr": 20,
        "gasch1": 42,
        "gasch2": 123,
        "seq": 13,
        "spo": 115,
    }
    epochss_GO = {
        "cellcycle": 62,
        "derisi": 91,
        "eisen": 123,
        "expr": 70,
        "gasch1": 122,
        "gasch2": 177,
        "seq": 45,
        "spo": 103,
    }
    epochss_others = {"diatoms": 474, "enron": 133, "imclef07a": 592, "imclef07d": 588}
    epochss = {"FUN": epochss_FUN, "GO": epochss_GO, "others": epochss_others}

    # Set seed
    seed = args.seed
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

    # Load the datasets
    if "others" in args.dataset:
        train, test = initialize_other_dataset(dataset_name, datasets)
    else:
        train, val, test = initialize_dataset(dataset_name, datasets)

    # Set the hyperparameters
    from utils.arff_reader import hierarchy_sizes

    best_model_path = "models/best_model.pth"
    batch_size = 24
    dropout = 0.6
    beta = 0.5
    relu_size = 384
    num_epochs = epochss[ontology][data]
    hierarchy = hierarchy_sizes.copy()
    while hierarchy and hierarchy[-1] == 0:
        hierarchy.pop()
    print(f"hierarchy: {hierarchy}, sum={sum(hierarchy)}")

    # Create loaders
    train_dataset = [(torch.tensor(row[:-1]).to(device), torch.tensor(row[-1]).to(device)) for row in train]
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    if "others" not in args.dataset:
        val_dataset = [(torch.tensor(row[:-1]).to(device), torch.tensor(row[-1]).to(device)) for row in val]
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        # train_dataset.extend(val_dataset)
    test_dataset = [(torch.tensor(row[:-1]).to(device), torch.tensor(row[-1]).to(device)) for row in test]
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Create the model
    model = HMCNFModel(
        input_dims[data],
        output_dims[ontology][data],
        hierarchy,
        beta=beta,
        dropout_rate=dropout,
        relu_size=relu_size,
    )

    # Initialize the model
    model.to(device)
    if device.index == 0 or device.type == "cpu":
        summary(model, input_size=(input_dims[data],), device=device.type)
    if next(model.parameters(), None) is not None:
        print("Model on gpu", next(model.parameters()).is_cuda)
    else:
        print("Model has no parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCELoss()

    # Train the model
    best_val_loss = float("inf")
    with tqdm(total=num_epochs * len(train_loader)) as pbar:
        for epoch in range(num_epochs):
            # Train
            model.train()
            for i, (x, y) in enumerate(train_loader):
                x = x.to(device).float()
                y = y.to(device).float()

                y_pred = model(x)
                # Debugging: Print shapes and ranges
                # print(f"Input shape: {x.shape}, Target shape: {y.shape}")
                # print(f"Input range: {x.min().item()} - {x.max().item()}, Target range: {y.min().item()} - {y.max().item()}")

                y_pred = model(x)
                # Debugging: Print model output range
                # print(f"Model output range: {y_pred.min().item()} - {y_pred.max().item()}")
                loss = criterion(y_pred, y)
                pbar.set_description(
                    f"Epoch: [{epoch + 1}/{num_epochs}], Step: [{i+1}/{len(train_loader)}] | train loss: {loss.data.cpu().numpy():.4f}, val loss: {best_val_loss:.4f}"
                )
                pbar.update()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Validate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.to(device).float()
                    y_val = y_val.to(device).float()

                    y_val_pred = model(x_val)
                    val_loss += criterion(y_val_pred, y_val).item()
            val_loss /= len(val_loader)
            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)

    # Load the best model
    model.load_state_dict(torch.load(best_model_path))
    # Test the model
    for i, (x, y) in enumerate(test_loader):

        model.eval()

        x = x.to(device).float()
        y = y.to(device).float()

        constrained_output = model(x)
        predicted = constrained_output.data > 0.5

        # Move output and label back to cpu to be processed by sklearn
        predicted = predicted.to("cpu")
        cpu_constrained_output = constrained_output.to("cpu")
        y = y.to("cpu")

        if i == 0:
            predicted_test = predicted
            constr_test = cpu_constrained_output
            y_test = y
        else:
            predicted_test = torch.cat((predicted_test, predicted), dim=0)
            constr_test = torch.cat((constr_test, cpu_constrained_output), dim=0)
            y_test = torch.cat((y_test, y), dim=0)

    score = average_precision_score(y_test, constr_test.data, average="micro")

    print(f"Average Precision Score: {score}")

    os.makedirs("results", exist_ok=True)
    f = open("results/" + dataset_name + ".csv", "a")
    f.write(str(seed) + "," + str(epoch) + "," + str(score) + "\n")
    f.close()


if __name__ == "__main__":
    main()
