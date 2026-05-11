from dataset import Mayo_Dataset
from torch.utils.data import DataLoader
from util import transforms
from util.util import CharbonnierLoss
import torch
from tester import test
from options.test_options import TestOptions
from litformer import LITFormer


if __name__ == '__main__':

    min_value = -1000
    max_value = 2000

    val_raw_transformer = transforms.Compose([
        transforms.Normalize(min_value=min_value, max_value=max_value),
        transforms.ToTensor(expand_dims=False)
    ])

    val_label_transformer = transforms.Compose([
        transforms.Normalize(min_value=min_value, max_value=max_value),
        transforms.ToTensor(expand_dims=False)
    ])

    val_transforms = [val_raw_transformer, val_label_transformer]

    opt = TestOptions().parse()
    device = torch.device(
        f'cuda:{opt.gpu_ids[0]}' if torch.cuda.is_available() else 'cpu'
    )

    print(">>> Testing model:", opt.model_path)

    test_dataset = Mayo_Dataset(opt, transforms=val_transforms)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=opt.test_batch_size,
        shuffle=False,
        num_workers=0
    )

    model = LITFormer(
        in_channels=1,
        out_channels=1,
        n_channels=64,
        num_heads_s=[1, 2, 4, 8],
        num_heads_t=[1, 2, 4, 8],
        res=True,
        attention_s=True,
        attention_t=True
    ).to(device)

    if len(opt.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    model.load_state_dict(
        torch.load(opt.model_path, map_location=device),
        strict=False
    )

    loss_fn = CharbonnierLoss()

    test(
        opt=opt,
        model=model,
        loss_fn=loss_fn,
        testloader=test_dataloader,
        device=device
    )
