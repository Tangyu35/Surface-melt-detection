import os
import shutil
import time
import json

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
from src import deeplabv3_resnet50, deeplabv3plus_mobilenetv3_large, deeplabv3plus_resnet50, deeplabv3plus_mobilenetv3_large_with_cbam, deeplabv3plus_resnet50_with_cbam


def mk_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def get_model_info(weights_path):
    # default = "./weights/mobilenet/with_cbam/Norm/norm_mobilenet_model_99.pth"
    model_backbone = weights_path.split('/')[2]
    use_cbam = True if 'without' not in weights_path.split('/')[3] else False
    use_norm = True if weights_path.split('/')[4] == 'Norm' else False

    return model_backbone, use_cbam, use_norm

def get_model_from_args(model_backbone, use_cbam, num_classes, aux, use_norm=True):
    if model_backbone == 'mobilenet':
        if use_cbam:
            return deeplabv3plus_mobilenetv3_large_with_cbam(aux=aux, num_classes=num_classes)
        else:
            return deeplabv3plus_mobilenetv3_large(aux=aux, num_classes=num_classes)

    if model_backbone == 'resnet':
        if use_cbam:
            return deeplabv3plus_resnet50_with_cbam(aux=aux, num_classes=num_classes)
        else:
            return deeplabv3plus_resnet50(aux=aux, num_classes=num_classes)

    exit(-1)

def main():
    aux = False  # inference time not need aux_classifier
    num_classes = 1
    weights_path = "./weights/resnet/without_cbam/Norm/norm_resnet_model_99.pth"
    # weights_path = "./weights/mobilenet/Norm/mobilenet_model_99.pth"
    VOC_path = r"E:\IRSA\Ice_Shelf\DLCode\deep-learning-for-image-processing-master\pytorch_segmentation\unet\snowmelt_dataset\VOCdevkit"
    image_folder_path = os.path.join(VOC_path, "VOC2012", "JPEGImagesFPS")

    palette_path = "./palette.json"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(image_folder_path), f"image {image_folder_path} not found."
    assert os.path.exists(palette_path), f"palette {palette_path} not found."
    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    model_backbone, use_cbam, use_norm = get_model_info(weights_path)
    num_classes = num_classes + 1
    model = get_model_from_args(model_backbone, use_cbam, num_classes, aux, use_norm=True)

    model_info = (
        'backbone: {}\n'
        'use_norm: {}\n'
        'use_cbam: {}\n'
        ).format(
        model_backbone,
        use_norm,
        use_cbam,
        )
    print("model info {}".format(model_info))
    # delete weights about aux_classifier
    weights_dict = torch.load(weights_path, map_location='cpu')['model']
    for k in list(weights_dict.keys()):
        if "aux" in k:
            del weights_dict[k]

    # load weights
    model.load_state_dict(weights_dict)
    model.to(device)
    predict_speed = []

    # model Para. count
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params / 1e6)

    for image_name in tqdm(os.listdir(image_folder_path)):
        img_path = os.path.join(image_folder_path, image_name)
        # load image
        original_img = Image.open(img_path).convert('RGB')

        # from pil image to tensor and normalize
        data_transform = transforms.Compose([transforms.Resize(520),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                  std=(0.229, 0.224, 0.225))])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()  # 进入验证模式
        with torch.no_grad():
            # init model
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            output = model(img.to(device))
            t_end = time_synchronized()
            predict_speed.append(t_end - t_start)
            # print("inference+NMS time: {}".format(t_end - t_start))

            prediction = output['out'].argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            mask = Image.fromarray(prediction)
            mask.putpalette(pallette)
            mask.save(os.path.join(out_path, image_name.replace('.jpg', '.png')))
    # model fps count
    FPS = 1 / (sum(predict_speed) / len(predict_speed))
    print("FPS: {}".format(FPS))

if __name__ == '__main__':
    out_path = './resnet_predict_result'
    mk_dir(out_path)

    main()
