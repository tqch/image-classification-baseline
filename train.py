if __name__ == "__main__":
    import os
    import json
    from train_utils import build_all
    import torch
    import torch.nn as nn
    from tqdm import tqdm

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--chkpt-dir", type=str, default="./chkpts")
    parser.add_argument("--config-dir", type=str, default="./configs")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--chkpt-intv", type=int, default=10)

    args = parser.parse_args()

    dataset = args.dataset

    config_path = os.path.join(args.config_dir, dataset + ".json")

    with open(config_path, "r") as f:
        configs = json.load(f)
        mod_configs = configs["model"]
        mod_name = args.model_name or mod_configs.pop("name", None)
        if mod_name:
            mod_configs = None
        opt_configs = configs["optimizer"]
        sch_configs = configs["scheduler"]
        dat_configs = configs["data"]

    model, optimizer, scheduler, trainloader, testloader = build_all(
        mod_name, mod_configs, opt_configs, sch_configs, dat_configs)

    del dat_configs["dataset"]

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    model_name = mod_name or model.__class__.__name__.lower()
    model.to(device)

    print(f"Dataset: {dataset}")
    print(f"Model: {model_name}")
    print(f"Optimizer setting: {opt_configs}")
    print(f"Scheduler setting: {sch_configs}")
    print(f"Dataloader setting: {dat_configs}")

    criterion = nn.CrossEntropyLoss(reduction="mean")

    chkpt_intv = args.chkpt_intv
    chkpt_dir = args.chkpt_dir
    chkpt_path = os.path.join(chkpt_dir, f"{dataset}_{model_name}.pt")
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)

    print(f"Save checkpoint every {chkpt_intv} epochs.")
    print(f"Checkpoint will be saved to {os.path.abspath(chkpt_path)}")

    def resume(path, model, optimizer, scheduler):
        model_dict = torch.load(path, map_location=device)
        model.load_state_dict(model_dict["model"])
        optimizer.load_state_dict(model_dict["optimizer"])
        scheduler.load_state_dict(model_dict["scheduler"])
        return model_dict["epoch"]

    start_epoch = 0
    if args.resume:
        print("Resuming training from checkpoint...")
        if os.path.exists(chkpt_path):
            try:
                start_epoch = resume(chkpt_path, model, optimizer, scheduler)
                print("Checkpoint has been loaded successfully!")
            except RuntimeError:
                print("Checkpoint fails to be loaded! Please check again!")
        else:
            print("Checkpoint does not exist!")

    epochs = configs["epochs"]
    print("Training starts...")

    for e in range(start_epoch, epochs):
        model.train()
        total_train_loss = 0
        total_train_correct = 0
        total_train_count = 0
        total_test_loss = 0
        total_test_correct = 0
        total_test_count = 0
        with tqdm(trainloader, desc=f"{e + 1}/{epochs} epochs") as t:
            for i, (x, y) in enumerate(t):
                out = model(x.to(device))
                loss = criterion(out, y.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pred = out.max(dim=-1)[1]
                total_train_loss += loss.item() * x.shape[0]
                total_train_correct += (pred == y.to(device)).sum().item()
                total_train_count += x.shape[0]
                if i == len(trainloader) - 1:
                    model.eval()
                    with torch.no_grad():
                        for x, y in testloader:
                            out = model(x.to(device))
                        pred = out.max(dim=-1)[1]
                        total_test_loss += loss.item() * x.shape[0]
                        total_test_correct += (pred == y.to(device)).sum().item()
                        total_test_count += x.shape[0]
                    t.set_postfix({
                        "train_loss": total_train_loss / total_train_count,
                        "train_acc": total_train_correct / total_train_count,
                        "test_loss": total_test_loss / total_test_count,
                        "test_acc": total_test_correct / total_test_count
                    })
                else:
                    t.set_postfix({
                        "train_loss": total_train_loss / total_train_count,
                        "train_acc": total_train_correct / total_train_count
                    })
            scheduler.step()
            if (e + 1) % chkpt_intv == 0:
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": e + 1
                }, chkpt_path)
