import argparse
import os
import time
import torch
from torch.utils.data import DataLoader

torch.cuda.set_device(0)
import dataset
import loss
import  models
from ssimloss import ssim
from collections import OrderedDict


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = self.args.device
        self.gnet = models.Generator()
        self.dnet = models.Discriminator()
        batch = self.args.batch
        self.train_loader = DataLoader(dataset.SRSFGANDataset(root=self.args.data_path, mode="train"),
                                       batch_size=batch, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(dataset.SRSFGANDataset(root=self.args.data_path, mode="val"),
                                     batch_size=1, shuffle=False, drop_last=True)
        self.ContentLoss = loss.ReconstructionLoss()
        self.AdversarialLoss = loss.AdversarialLoss()
        self.Perceptualloss = loss.ContentLoss(self.device)
        self.mse = torch.nn.L1Loss()
        self.criterion_d = torch.nn.BCELoss()
        self.sobelloss = loss.ReconstructionLoss()
        self.epoch = 0
        self.lr = 1e-3*2
        self.best_psnr = 0.
        if self.args.resume:
            if not os.path.exists(self.args.save_path):
                print("No params, start training...")
            else:
                param_dict = torch.load(self.args.save_path)
                dnet_param = param_dict["dnet_dict"]
                new_state_dict = OrderedDict()
                for k, v in dnet_param.items():
                    name = k[7:]
                    new_state_dict[name] = v
                dnet_param = new_state_dict
                gnet_param = param_dict["gnet_dict"]
                new_state_dict = OrderedDict()
                for k, v in gnet_param.items():
                    name = k[7:]
                    new_state_dict[name] = v
                gnet_param = new_state_dict
                self.epoch = param_dict["epoch"]
                self.lr = param_dict["lr"]
                self.dnet.load_state_dict(dnet_param)
                self.gnet.load_state_dict(gnet_param)
                self.best_psnr = param_dict["best_psnr"]
                print("Loaded params from {}\n[Epoch]: {}   [lr]: {}    [best_psnr]: {}".format(self.args.save_path,
                                                                                                self.epoch, self.lr,
                                                                                                self.best_psnr))
        device_ids = [0, 1, 2]
        self.dnet = torch.nn.DataParallel(self.dnet, device_ids=device_ids).cuda()
        self.gnet = torch.nn.DataParallel(self.gnet, device_ids=device_ids).cuda()
        self.gnet.cuda()

        self.optimizer_d = torch.optim.Adam(self.dnet.parameters(), lr=self.lr)
        self.optimizer_g = torch.optim.Adam(self.gnet.parameters(), lr=self.lr * 0.1)
        self.mse = torch.nn.L1Loss()
        self.cri = torch.nn.MSELoss()
        self.real_label = torch.ones([batch, 1, 256, 256]).to(self.device)
        self.real_label_val = torch.ones([1, 1, 256, 256]).to(self.device)
        self.fake_label = torch.zeros([batch, 1, 256, 256]).to(self.device)

    @staticmethod
    def calculate_psnr(img1, img2):
        return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

    def train(self, epoch):
        self.dnet.train()
        self.gnet.train()
        train_loss_d = 0.
        train_loss_g = 0.
        train_loss_all_d = 0.
        train_loss_all_g = 0.
        psnr = 0.
        total = 0
        start = time.time()
        print("Start epoch: {}".format(epoch))
        for i, (lt0s,mt1s,lt1s) in enumerate(self.train_loader):
            lt0s = lt0s.to(self.device)
            mt1s = mt1s.to(self.device)
            lt1s = lt1s.to(self.device)
            label = lt1s.to(self.device)
            self.dnet.zero_grad()
            self.gnet.zero_grad()
            fake_img,outs = self.gnet(lt0s, mt1s)
            real_out = self.dnet(label)
            fake_out = self.dnet(fake_img.detach())
            loss_d = 0.5 * torch.mean((real_out - self.real_label) ** 2) + 0.5 * torch.mean(
                (fake_out - self.fake_label) ** 2)

            loss_d.backward()
            self.optimizer_d.step()
            train_loss_d += loss_d.item()
            train_loss_all_d += loss_d.item()
            loss_g =  0.01*self.AdversarialLoss(self.dnet(fake_img),self.real_label)+self.mse(fake_img,label) +self.sobelloss(fake_img,label)+self.sobelloss(outs,label)#+ 1.0-torch.mean(F.cosine_similarity(fake_img, label, 1))
            self.optimizer_g.zero_grad()
            loss_g.backward()
            self.optimizer_g.step()
            train_loss_g += loss_g.item()
            train_loss_all_g += loss_g.item()
            psnr += self.calculate_psnr(fake_img, label).item()
            total += 1

            if (i + 1) % self.args.interval == 0:
                end = time.time()
                print("[Epoch]: {}[Progress: {:.1f}%]time:{:.2f} gnet_loss:{:.5f} psnr:{:.4f}".format(
                    epoch, (i + 1) * 100 / len(self.train_loader), end - start,

                           train_loss_g / self.args.interval, psnr / total
                ))
                train_loss_d = 0.
                train_loss_g = 0.
        print("Save params to {}".format(self.args.save_path1))
        param_dict = {
            "epoch": epoch,
            "lr": self.lr,
            "best_psnr": self.best_psnr,
            "dnet_dict": self.dnet.state_dict(),
            "gnet_dict": self.gnet.state_dict()
        }
        torch.save(param_dict, self.args.save_path)
        return  train_loss_all_d/ len(self.train_loader),train_loss_all_g / len(self.train_loader), psnr / total

    def val(self, epoch):
        self.gnet.eval()
        self.dnet.eval()
        print("Test start...")
        val_loss = 0.
        psnr = 0.
        total = 0
        start = time.time()
        with torch.no_grad():
            for i, (lt0s,mt1s,lt1s) in enumerate(self.val_loader):
                lt0s = lt0s.to(self.device)
                mt1s = mt1s.to(self.device)
                lt1s = lt1s.to(self.device)
                label = lt1s.to(self.device)
                fake_img,outs = self.gnet(lt0s, mt1s)
                loss =   self.ContentLoss(fake_img,label)
                val_loss += loss.item()
                psnr += self.calculate_psnr(fake_img, label).item()
                total += 1
            if psnr == 0:
                mpsnr = 0
            else:
                mpsnr = psnr / total
            end = time.time()
            print("Test finished!")
            print("[Epoch]: {} time:{:.2f} loss:{:.5f} ssim:{:.4f}".format(
                epoch, end - start, val_loss / len(self.val_loader), mpsnr
            ))
            if mpsnr > self.best_psnr:
                self.best_psnr = mpsnr
                print("Save params to {}".format(self.args.save_path))
                param_dict = {
                    "epoch": epoch,
                    "lr": self.lr,
                    "best_psnr": self.best_psnr,
                    "gnet_dict": self.gnet.state_dict(),
                    "dnet_dict": self.dnet.state_dict()
                }
                torch.save(param_dict, self.args.save_path1)
            print(ssim(fake_img, label))
        return val_loss / len(self.val_loader), mpsnr




def main(args):
    t = Trainer(args)
    for epoch in range(t.epoch, t.epoch + args.num_epochs):
        train_loss_d,train_loss_g, train_psnr = t.train(epoch)
        val_loss, val_psnr = t.val(epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training SRGAN with celebA")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--data_path", default=r"traindata", type=str)
    parser.add_argument("--resume", default=True, type=bool)
    parser.add_argument("--num_epochs", default=160, type=int)
    parser.add_argument("--save_path", default=r"./qingzanglake00.pt", type=str)
    parser.add_argument("--save_path1", default=r"./qingzanglake01.pt", type=str)
    parser.add_argument("--interval", default=20, type=int)
    parser.add_argument("--batch", default=16, type=int)
    args1 = parser.parse_args()
    main(args1)
=")
        print("Learning rate has adjusted to {}".format(self.lr))


def main(args):
    t = Trainer(args)
    for epoch in range(t.epoch, t.epoch + args.num_epochs):
        train_loss_d,train_loss_g, train_psnr = t.train(epoch)
        val_loss, val_psnr = t.val(epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training SRGAN with celebA")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--data_path", default=r"traindata", type=str)
    parser.add_argument("--resume", default=True, type=bool)
    parser.add_argument("--num_epochs", default=160, type=int)
    parser.add_argument("--save_path", default=r"./qingzanglake00.pt", type=str)
    parser.add_argument("--save_path1", default=r"./qingzanglake01.pt", type=str)
    parser.add_argument("--interval", default=20, type=int)
    parser.add_argument("--batch", default=16, type=int)
    args1 = parser.parse_args()
    main(args1)
