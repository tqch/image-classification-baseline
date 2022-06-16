import os
import json
from train_utils import build_model_from_name, build_dataloader
import torch
from tqdm import tqdm

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--chkpt-dir", type=str, default="./chkpts")
    parser.add_argument("--config-dir", type=str, default="./configs")
    parser.add_argument("--device", type=str, default="0")

    args = parser.parse_args()

    dataset = args.dataset
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    mod_name = args.model_name
    config_path = os.path.join(
        args.config_dir, f"{dataset}{'_' + mod_name if mod_name else ''}.json")

    if not os.path.exists(config_path):
        print("Model-specific config file not found!")
        print("Falling back to common config...")
        config_path = os.path.join(args.config_dir, f"{dataset}.json")

    with open(config_path, "r") as f:
        configs = json.load(f)
    mod_configs = configs["model"]
    model_name = mod_name or mod_configs.pop("name", None)

    print(f"Dataset: {dataset}")
    print(f"Model: {model_name}")
    print(f"Constructing model...")
    model = build_model_from_name(model_name)
    model.to(device)
    print(f"Loading trained weights...")
    model.load_state_dict(torch.load(f"chkpts/{dataset}_{model_name}.pt", map_location=device)["model"])
    model.eval()

    trainloader, testloader = build_dataloader({"dataset": dataset, "batch_size": 256})

    print("Evaluation starts...", flush=True)

    tot_corr = 0
    tot_cnt = 0
    with torch.no_grad():
        for x, y in tqdm(testloader):
            pred = model(x.to(device)).max(dim=1)[1]
            tot_corr += (pred.cpu() == y).sum().item()
            tot_cnt += x.shape[0]
    print("Test accuracy is {}".format(tot_corr / tot_cnt))
