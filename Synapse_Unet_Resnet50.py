import sys
import os
sys.path.append(os.path.abspath('/data3/nkozah/my_project'))
from networks_ibrahim import A1215_UNet_Resnet50_binary_base_pretrain_Multilabel as SegNet
from tqdm import tqdm
import utils
import random
import argparse
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.utils.data import DataLoader
from skimage import segmentation
import copy
from scipy import stats
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
from losses.DiceLoss import MultiClassDiceLoss
from metrics.MultiLabel_metric import multilabel_metric
import cv2
import torch
import torch.nn as nn
import datetime

torch.cuda.set_device(0)

def make_print_to_file(path="./logs"):
    """
    Redirect all print() output to a dated log file in `path`,
    while still printing to the console.
    """
    import os
    import sys
    import datetime

    os.makedirs(path, exist_ok=True)  # <-- IMPORTANT

    class Logger(object):
        def __init__(self, filename="Default.log", path="."):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding="utf8")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    fileName = datetime.datetime.now().strftime("DAY%Y_%m_%d_")
    sys.stdout = Logger(fileName + "Synapse_UNet_Resnet50Encoder.log", path=path)
make_print_to_file(path='./logs')

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='Synapse', help='experiment_name')
    parser.add_argument("--train_dataset", type=str, default='/Data/synapse/train_npz',
                        help="path to Dataset")
    parser.add_argument('--root_path_train', type=str,
                        default='/Data/synapse/train_npz', help='root_path_train')
    parser.add_argument('--list_dir', type=str,
                        default='/Data/synapse/lists_Synapse', help='list dir')
    parser.add_argument('--num_classes', type=int,
                        default=9, help='output channel of network')
    parser.add_argument("--patience", type=int, default=50,
                        help="patience epoch number (default: 30k)")
    # Train Options
    parser.add_argument("--RESUME", type=bool, default=False)
    parser.add_argument("--START_EPOCH", type=int, default=0,
                        help="epoch number (default: 30k)")
    parser.add_argument("--NB_EPOCH", type=int, default=230,
                        help="epoch number (default: 30k)")
    parser.add_argument("--warmup_itrs", type=int, default=5,
                        help="epoch number (default: 1k)")
    parser.add_argument("--LR", type=float, default=0.001,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=True, #修改！！！
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=4,
                        help='batch size (default: 24)')
    parser.add_argument("--val_batch_size", type=int, default=12,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--img_size", type=int, default=224) #修改！！！

    parser.add_argument("--gpu_id", type=str, default='0',help="GPU ID")
    parser.add_argument("--gpu_ids", type=list, default=[0],help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1234,
                        help="random seed (default: 1)")
    #parser.add_argument("--beta_cutmix", type=float, default=0.3)
    #parser.add_argument("--cutmix_prob", type=float, default=1)
    #parser.add_argument("--N_min", type=int, default=200)
    #parser.add_argument("--N_max", type=int, default=400)

    #parser.add_argument("--model_name", type=str, default='hiformer-b', help='model name')


    return parser

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes



results_out_dir = './Results_out/'
if not os.path.exists(results_out_dir):
    os.makedirs(results_out_dir)

def main():
    opts = get_argparser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    tr_transform= RandomGenerator(output_size=[opts.img_size, opts.img_size])

    train_dataset = Synapse_dataset(base_dir=opts.root_path_train, list_dir=opts.list_dir, split="train",
                               transform=tr_transform)
    print("The length of train set is: {}".format(len(train_dataset)))

    train_loader = DataLoader(train_dataset,
                              batch_size=opts.batch_size,
                              shuffle=True)
    print("Train: ", len(train_dataset))


    model_name = './checkpoints/Synapse_UNet_Resnet50Encoder.pth'
    
    if len(opts.gpu_ids) > 1:
        if opts.RESUME:
            model = SegNet.UNetWithResnet50Encoder(n_classes=opts.num_classes).cuda()
            model.load_state_dict(torch.load(model_name))
            model = model.cuda()
        else:
            model = SegNet.UNetWithResnet50Encoder(n_classes=opts.num_classes).cuda()
            model = model.cuda()
        model = torch.nn.DataParallel(model)
    else:
        if opts.RESUME:
            model= SegNet.UNetWithResnet50Encoder(n_classes=opts.num_classes).cuda()
            model.load_state_dict(torch.load(model_name))
            model = model.cuda()
        else:
            model = SegNet.UNetWithResnet50Encoder(n_classes=opts.num_classes).cuda()
            model = model.cuda()
 
    criterion_seg = nn.CrossEntropyLoss(reduction='mean')
    criterion_dice = DiceLoss(n_classes=opts.num_classes)

    tm = datetime.datetime.now().strftime('T' + '%m%d%H%M')
    results_file_name = results_out_dir + tm + 'Synapse_UNet_Resnet50Encoder_results.txt'

    best_dice, best_iou = 0.0, 0.0
    for epoch in range(opts.START_EPOCH, opts.NB_EPOCH):
        if epoch <= 4:
            if opts.LR >= 0.00001:
                warmup_lr = opts.LR * ((epoch + 1) / 5)
                lr = warmup_lr
            else:
                lr = opts.LR
        elif 4 < epoch <= 149:
            lr = opts.LR
        elif 149 < epoch <= 169:  # 50
            lr = opts.LR / 2
        elif 169 < epoch <= 189:  # 40
            lr = opts.LR / 4
        elif 189 < epoch <= 209:  # 30
            lr = opts.LR / 8
        elif 209 < epoch <= 219:  # 40
            lr = opts.LR / 10
        elif 219 < epoch <= 230:  # 40
            lr = opts.LR / 100
        
        print("current epoch:", epoch, "current lr:", lr)

        optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=opts.weight_decay)

        list_loss, list_loss_seg, list_loss_dice = [], [], []
        best_val_loss = 1000
        for i, samples in tqdm(enumerate(train_loader)):
            images = samples['image']
            labels = samples['label']

            if images.size()[1] == 1:
                images = images.repeat(1, 3, 1, 1)
 
            optimizer.zero_grad()
            imgs_aug = images.to(device, dtype=torch.float32)
            lbs_aug = labels.to(device, dtype=torch.long)
            model.train()
            out_seg = model(imgs_aug)
            loss_seg = criterion_seg(out_seg, lbs_aug)
            loss_dice = criterion_dice(out_seg, lbs_aug)
            loss = loss_seg + loss_dice

            list_loss.append(loss)
            list_loss_seg.append(loss_seg)
            list_loss_dice.append(loss_dice)

            # print("loss: ", loss)
            # print("loss_seg: ", loss_seg)
            # print("loss_dice: ",  + loss_dice)

            
            loss.backward()
            optimizer.step()

        if len(list_loss) > 0:
            list_loss = torch.stack(list_loss).mean()
        else:
            list_loss = 0

        if len(list_loss_seg) > 0:
            list_loss_seg = torch.stack(list_loss_seg).mean()
        else:
            list_loss_seg = 0

        if len(list_loss_dice) > 0:
            list_loss_dice = torch.stack(list_loss_dice).mean()
        else:
            list_loss_dice = 0

        torch.save(model.state_dict(), model_name)
        if list_loss < best_val_loss:
            best_val_loss = list_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= opts.patience:
            print(f"Early stopping after {epoch + 1} epochs.")
            break
        print(
            "Epoch %d, Loss=%f, Loss_SEG = %f, Loss_DICE = %f" %
            (epoch, list_loss, list_loss_seg, list_loss_dice))

        with open(results_file_name, 'a') as file:
            file.write(
                'Epoch %d, Loss=%f, Loss_SEG=%f, Loss_DICE=%f \n ' % (
                    epoch, list_loss, list_loss_seg, list_loss_dice))

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()