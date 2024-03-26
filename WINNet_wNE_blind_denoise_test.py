import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.networks import WINNetklvl, noiseEst
from utils.dataset import *
from PIL import Image

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser(description="WINNet with Noise Estimation for Blind Denoise")
parser.add_argument("--num_of_steps", type=int, default=4, help="Number of steps")
parser.add_argument("--num_of_layers", type=int, default=4, help="Number of layers")
parser.add_argument("--num_of_channels", type=int, default=32, help="Number of channels")
parser.add_argument("--lvl", type=int, default=1, help="number of levels")
parser.add_argument("--split", type=str, default="dct", help='splitting operator')
parser.add_argument("--dnlayers", type=int, default=4, help="Number of denoising layers")
parser.add_argument("--logdirdn", type=str, default="logs", help='path of log files')
parser.add_argument("--logdirne", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
parser.add_argument("--show_results", type=bool, default=True, help="show results")

opt = parser.parse_args()


def main():
    if not os.path.exists('results/WINNet_blind'):
        os.makedirs('results/WINNet_blind')
    if not os.path.exists('results/Noisy_blind'):
        os.makedirs('results/Noisy_blind')

    # Build model
    device_ids = [0]

    dnnet = WINNetklvl(steps=opt.num_of_steps, layers=opt.num_of_layers, channels=opt.num_of_channels, klvl=opt.lvl,
                     mode=opt.split, dnlayers=opt.dnlayers)
    dnmodel = nn.DataParallel(dnnet, device_ids=device_ids).cuda()

    nenet = noiseEst(psz=8, stride=1, num_layers=5, f_ch=16, fsz=5)
    nemodel = nn.DataParallel(nenet, device_ids=device_ids).cuda()

    torch.manual_seed(123)

    dnmodel.load_state_dict(torch.load(os.path.join(opt.logdirdn, 'net_WINNet.pth')))
    nemodel.load_state_dict(torch.load(os.path.join(opt.logdirne, 'net_NENet.pth')))

    pytorch_total_params = sum(p.numel() for p in dnmodel.parameters() if p.requires_grad)
    print('Total Number of Parameters: ', pytorch_total_params)

    dnmodel.eval()
    nemodel.eval()
    # load data info
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
    files_source.sort()
    # process data
    psnr_avg = 0
    nlvl_avg = 0
    i = 1
    for f in files_source:
        # image
        Img = cv2.imread(f)
        Img = normalize(np.float32(Img))
        if len(Img.shape) == 2: 
            channels = [Img]
        else: 
            R, G, B = cv2.split(Img)
            channels = [R, G, B]
        Out_channels = []
        ISource_channels = []
        INoisy_channels = []
        for ch in channels:
            ch = np.expand_dims(ch, 0)
            ch = np.expand_dims(ch, 1)
            ISource = torch.Tensor(ch)
            if ISource.size(2) % 2 == 1:
                ISource = ISource[:, :, 0:ISource.size(2) - 1, :]
            if ISource.size(3) % 2 == 1:
                ISource = ISource[:, :, :, 0:ISource.size(3) - 1]

            # noise
            noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL / 255.)
            # noisy image
            INoisy = ISource + noise
            ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())

            with torch.no_grad():
                nEst, _, _ = nemodel(INoisy)
                nEst = Variable(nEst * torch.ones(1).cuda())
                out, _ = dnmodel(INoisy, nEst)
                out = torch.clamp(out, 0., 1.)
            Out_channels.append(out.cpu().detach().numpy().squeeze())
            ISource_channels.append(ISource.cpu().detach().numpy().squeeze())
            INoisy_channels.append(INoisy.cpu().detach().numpy().squeeze())

        Out = cv2.merge(Out_channels)
        ISource = cv2.merge(ISource_channels)
        INoisy = cv2.merge(INoisy_channels)
        Out_tensor = torch.tensor(Out, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        ISource_tensor = torch.tensor(ISource, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        INoisy_tensor = torch.tensor(INoisy, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        psnr = batch_PSNR(Out_tensor, ISource_tensor, data_range=1., crop=0)

        if opt.show_results:
            save_out_path = "results/WINNet_blind/out_img_{}.png".format(i)
            save_img(save_out_path, Out_tensor)
            save_out_path = "results/Noisy_blind/out_img_{}.png".format(i)
            save_img(save_out_path, INoisy_tensor)
        i += 1

        psnr_avg += psnr
        nlvl_avg += nEst

    psnr_avg /= len(files_source)
    nlvl_avg /= len(files_source)
    print('Average PSNR: %f'% psnr_avg)
    print('Average estimated noise level: %f' % nlvl_avg)

if __name__ == "__main__":
    main()
