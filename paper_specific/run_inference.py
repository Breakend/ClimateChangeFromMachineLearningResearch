import torch
from thop.profile import profile
from PIL import Image
from torchvision import transforms
from experiment_impact_tracker.compute_tracker import ImpactTracker, get_flop_count_tensorflow
import argparse
import os
import random
import numpy as np

models = [ ('PingoLH/Pytorch-HarDNet', 'hardnet68'),
           ('PingoLH/Pytorch-HarDNet', 'hardnet85'),
           ('PingoLH/Pytorch-HarDNet', 'hardnet68ds'),
           ('PingoLH/Pytorch-HarDNet', 'hardnet39ds'),
           ('pytorch/vision', 'googlenet'),
           ('pytorch/vision', 'alexnet'),
           ('pytorch/vision', 'shufflenet_v2_x1_0'),
           ('pytorch/vision', 'vgg11'),
           ('pytorch/vision', 'vgg13'),
           ('pytorch/vision', 'vgg16'),
           ('pytorch/vision', 'vgg19'),
           ('pytorch/vision', 'mobilenet_v2'),
           ('pytorch/vision', 'densenet121'),
           ('pytorch/vision', 'densenet169'),
           ('pytorch/vision', 'densenet201'),
           ('pytorch/vision', 'densenet161'),
           ('pytorch/vision', 'squeezenet1_0'),
           ('pytorch/vision', 'squeezenet1_1'),
           ('pytorch/vision', 'wide_resnet50_2'),
           ('pytorch/vision', 'wide_resnet101_2'),
           ('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl'),
           ('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl'),
           ('facebookresearch/WSL-Images', 'resnext101_32x32d_wsl'),
           ('facebookresearch/WSL-Images', 'resnext101_32x48d_wsl'),
]

#host = os.environ['HOSTNAME'].replace(".stanford.edu","")
#os.environ['TORCH_HOME'] = "/{}/scr1/phend/".format(host)
#print("Using torchhome of {}".format("/{}/scr1/phend/".format(host)))
print("Running on machine: {}".format(os.environ["SLURM_SUBMIT_HOST"]))
os.environ['TORCH_HOME'] =  "/{}/scr1/phend/".format(os.environ["SLURM_SUBMIT_HOST"].replace(".stanford.edu", ""))
directory = "/{}/scr1/phend/".format(os.environ["SLURM_SUBMIT_HOST"].replace(".stanford.edu", ""))
import os
if not os.path.exists(directory):
    os.makedirs(directory)
# Open child processes via os.system(), popen() or fork() and execv()


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('model', type=int, help="Select model from constant list")
parser.add_argument('seed', type=int)
parser.add_argument('log_dir', type=str)

args = parser.parse_args()
np.random.seed(args.seed)
random.seed(args.seed)

if "resnext101" in models[args.model][1]:
    model = torch.hub.load(models[args.model][0], models[args.model][1])
else:
    model = torch.hub.load(models[args.model][0], models[args.model][1], pretrained=True)
model.eval()

input_image = Image.open("./dog.jpg")
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

total_ops, total_params = profile(model, (input_batch,), verbose=False)

print("%s | %s | %s" % ("Model", "Params(M)", "FLOPs(G)"))
print("---|---|---")
print("%s | %.2f | %.2f" % (models[args.model][1], total_params / (1000 ** 2), total_ops / (1000 ** 3)))

tracker = ImpactTracker(os.path.join(args.log_dir, "{}_seed{}".format(models[args.model][1], args.seed)))
tracker.launch_impact_monitor()

for i in range(50000):
    print(i)
    with torch.no_grad():
        output = model(input_batch)
