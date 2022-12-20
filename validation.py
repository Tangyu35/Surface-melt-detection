import os
import torch
import datetime

from src import deeplabv3_resnet50, deeplabv3plus_mobilenetv3_large, deeplabv3plus_resnet50, deeplabv3plus_mobilenetv3_large_with_cbam, deeplabv3plus_resnet50_with_cbam
from train_utils import evaluate
from my_dataset import VOCSegmentation
import transforms as T


def get_model_info(weights_path):
    # default = "./weights/mobilenet/with_cbam/Norm/norm_mobilenet_model_99.pth"
    model_backbone = weights_path.split('/')[2]
    use_cbam = True if 'without' not in weights_path.split('/')[3] else False
    use_norm = True if weights_path.split('/')[4] == 'Norm' else False

    return model_backbone, use_cbam, use_norm

def get_model_from_args(model_backbone, use_cbam, num_classes, use_norm=True):
    if model_backbone == 'mobilenet':
        if use_cbam:
            return deeplabv3plus_mobilenetv3_large_with_cbam(aux=args.aux, num_classes=num_classes)
        else:
            return deeplabv3plus_mobilenetv3_large(aux=args.aux, num_classes=num_classes)

    if model_backbone == 'resnet':
        if use_cbam:
            return deeplabv3plus_resnet50_with_cbam(aux=args.aux, num_classes=num_classes)
        else:
            return deeplabv3plus_resnet50(aux=args.aux, num_classes=num_classes)

    exit(-1)

class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
                T.RandomResize(base_size, base_size),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def main(args):
    model_backbone, use_cbam, use_norm = get_model_info(args.weights)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    assert os.path.exists(args.weights), f"weights {args.weights} not found."

    results_file = "./validation/{}_val_info.txt".format(model_backbone)

    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
    val_dataset = VOCSegmentation(args.data_path,
                                  year="2012",
                                  transforms=SegmentationPresetEval(520),
                                  txt_name="val.txt")

    num_workers = 8
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=32,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    model = get_model_from_args(model_backbone, use_cbam, num_classes=num_classes, use_norm=True)
    model.load_state_dict(torch.load(args.weights, map_location=device)['model'])
    model.to(device)

    confmat = evaluate(model, val_loader, device=device, num_classes=num_classes)
    val_info = str(confmat)
    model_info = (
        'backbone: {}\n'
        'use_cbam: {}\n'
        'use_norm: {}\n'
        ).format(
        model_backbone,
        use_cbam,
        use_norm,
        )
    # print(val_info)
    print(model_info + val_info)





def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch deeplabv3 validation")

    parser.add_argument("--data-path", default=r"E:\IRSA\Ice_Shelf\DLCode\deep-learning-for-image-processing-master\pytorch_segmentation\unet\snowmelt_dataset_unnorm", help="VOCdevkit root")
    parser.add_argument("--weights", default="./weights/resnet/with_cbam/UnNorm/nonorm_resnet_model_98.pth")
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--aux", default=True, type=bool, help="auxilier loss")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
