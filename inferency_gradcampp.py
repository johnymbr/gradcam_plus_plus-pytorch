import os
import PIL
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp

device = 'cuda' if torch.cuda.is_available() else 'cpu'

img_dir = 'images'
# img_name = 'collies.JPG'
# img_name = 'multiple_dogs.jpg'
# img_name = 'snake.JPEG'
img_name = 'water-bird.JPEG'
img_path = os.path.join(img_dir, img_name)

weights_dir = 'weights'
weights_name = 'XXXXX'
weights_path = os.path.join(weights_dir, weights_name)

grid_cam_dir = 'cam_images'

pil_img = PIL.Image.open(img_path)
pil_img

torch_img = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])(pil_img).to(device)
normed_torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None]

densenet = models.densenet121(pretrained=True)
densenet.load_state_dict(torch.load(weights_path))

configs = [
    dict(model_type='densenet', arch=densenet, layer_name='features_norm5')
]

for config in configs:
    config['arch'].to(device).eval()

cams = [
    [cls.from_config(**config) for cls in (GradCAM, GradCAMpp)]
    for config in configs
]

images = []
for gradcam, gradcam_pp in cams:
    # mask, _ = gradcam(normed_torch_img)
    # heatmap, result = visualize_cam(mask, torch_img)

    mask_pp, _ = gradcam_pp(normed_torch_img)
    heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)

    images.extend([torch_img.cpu(), heatmap_pp, result_pp])

grid_image = make_grid(images, nrow=5)
grid_image.save(grid_cam_dir + '/' + img_name)
