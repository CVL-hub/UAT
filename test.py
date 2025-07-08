import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from models.UAT import Network
from data import get_dataloader
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
from utils.utils import load_model_params


def test_model(test_loader, model):

    Sm = Smeasure()
    Em = Emeasure()
    Fm = Fmeasure()
    wFm = WeightedFmeasure()
    mae = MAE()

    model.eval()
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as pbar:
            for (image, gt, sal_f, _) in test_loader:
                image = image.cuda()
                gt = gt.numpy().astype(np.float32).squeeze()
                gt /= (gt.max() + 1e-8)

                sal_f = sal_f.cuda()
                _, _, _, res = model(x=image, ref_x=sal_f, y=None, training=False)

                res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)                        # 标准化处理,把数值范围控制到(0,1)

                Sm.step(pred=res*255, gt=gt*255)
                Em.step(pred=res*255, gt=gt*255)
                Fm.step(pred=res*255, gt=gt*255)
                wFm.step(pred=res*255, gt=gt*255)
                mae.step(pred=res*255, gt=gt*255)

                pbar.update()

            sm = Sm.get_results()["sm"]
            em = Em.get_results()["em"]
            fm = Fm.get_results()["fm"]
            wfm = wFm.get_results()["wfm"]
            mae = mae.get_results()["mae"]

        results = {
            "Smeasure": sm,
            "WeightedFmeasure": wfm,
            "MAE": mae,
            "adpEm": em["adp"],
            "meanEm": em["curve"].mean(),
            "maxEm": em["curve"].max(),
            "adpFm": fm["adp"],
            "meanFm": fm["curve"].mean(),
            "maxFm": fm["curve"].max()
        }
        print(results)
        file = open(".\results.txt", "a")
        file.write(opt.model_name + str(results) + '\n')
        file.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='UAT')
    parser.add_argument('--dim', type=int, default=64, help='dimension of our model')
    parser.add_argument('--imgsize', type=int, default=352, help='testing image size')
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=8, help='the number of workers in dataloader')
    parser.add_argument('--gpu_id', type=str, default='4', help='train use gpu')
    parser.add_argument('--data_root', type=str, default='./dataset/R2C7K', help='the path to put dataset')
    parser.add_argument('--save_root', type=str, default='./snapshot', help='the path to save model params and log')
    opt = parser.parse_args()
    print(opt)

    # load model 
    ref_model = Network(opt).cuda()
    params_path = os.path.join(opt.save_root, opt.model_name, 'Net_epoch_45.pth')
    ref_model = load_model_params(ref_model, params_path)

    # load data
    test_loader = get_dataloader(opt.data_root, opt.shot, opt.imgsize, opt.num_workers, mode='test')

    # processing
    scores = test_model(test_loader, ref_model)

    print(scores)
