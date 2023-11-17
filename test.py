import random
import argparse
import numpy as np
import zqypackage
import torch
from collections import OrderedDict

from models import Generator



def test(model, save_path,IMAGE_SIZE, PATCH_SIZE,landsat_dir,modis_dir,save_dir):#假设IMAGE_SIZE:[1200,1200] PATCH_SIZE:256
    new_state_dict = OrderedDict()
    loaded_model =torch.load(save_path)
    loaded_model = loaded_model['gnet_dict']
    for k, v in loaded_model.items():
        name = k[7:]
        new_state_dict[name] = v

    param_dict = torch.load(save_path)
    print("Load params from {}\n[Epoch]: {}|[lr]: {}|[best_psnr]: {}".format(save_path,
                                                                             param_dict["epoch"],
                                                                             param_dict["lr"],
                                                                             param_dict["best_psnr"]))
    model.load_state_dict(new_state_dict)
    model.to("cuda")
    # cur_result = {}
    model.eval()
    modis, _,  _, _ = zqypackage.read_image(modis_dir)
    modis = torch.from_numpy(modis.astype(np.float32)).mul_(0.0001)
    # print(modis.shape)
    landsat, band,  image_proj, image_geotrans = zqypackage.read_image(landsat_dir)
    landsat = torch.from_numpy(landsat.astype(np.float32)).mul_(0.0001)
    images = [landsat, modis]
    PATCH_STRIDE = PATCH_SIZE // 2
    end_h = (IMAGE_SIZE[0] - PATCH_STRIDE) // PATCH_STRIDE * PATCH_STRIDE
    end_w = (IMAGE_SIZE[1] - PATCH_STRIDE) // PATCH_STRIDE * PATCH_STRIDE
    end_h_md = (IMAGE_SIZE[0]//8 - PATCH_STRIDE//8) // (PATCH_STRIDE//8) * (PATCH_STRIDE//8)
    end_wmd = (IMAGE_SIZE[1]//8 - PATCH_STRIDE//8) // (PATCH_STRIDE//8) * (PATCH_STRIDE//8)
    h_index_list = [i for i in range(0, end_h, PATCH_STRIDE)]#[0,128,256,384,512,640,768,896]
    w_index_list = [i for i in range(0, end_w, PATCH_STRIDE)]#[0,128,256,384,512,640,768,896]
    if (IMAGE_SIZE[0] - PATCH_STRIDE) % PATCH_STRIDE != 0:
        h_index_list.append(IMAGE_SIZE[0] - PATCH_SIZE)#[0,128,256,384,512,640,768,896,944]
    if (IMAGE_SIZE[1] - PATCH_STRIDE) % PATCH_STRIDE != 0:
        w_index_list.append(IMAGE_SIZE[1] - PATCH_SIZE)#[0,128,256,384,512,640,768,896,944]
    h_index_list_md = [i for i in range(0, end_h_md, PATCH_STRIDE//8)]#[0,128,256,384,512,640,768,896]
    w_index_list_md = [i for i in range(0, end_wmd, PATCH_STRIDE//8)]#[0,128,256,384,512,640,768,896]
    if (IMAGE_SIZE[0]//8 - PATCH_STRIDE//8) % (PATCH_STRIDE//8) != 0:
        h_index_list_md.append(IMAGE_SIZE[0]//8 - PATCH_SIZE//8)#[0,128,256,384,512,640,768,896,944]
    if (IMAGE_SIZE[1]//8 - PATCH_STRIDE//8) % (PATCH_STRIDE//8) != 0:
        w_index_list_md.append(IMAGE_SIZE[1]//8 - PATCH_SIZE//8)#[0,128,256,384,512,640,768,896,944]

    output_image = np.zeros(images[0].shape)
    # print(output_image.shape)

    for i in range(len(h_index_list)):
        for j in range(len(w_index_list)):
            h_start = h_index_list[i]
            w_start = w_index_list[j]
            h_start_md = h_index_list_md[i]
            w_start_md = w_index_list_md[j]
            input_lr = images[1][:, h_start_md: h_start_md + PATCH_SIZE//8, w_start_md: w_start_md + PATCH_SIZE//8]
            # target_hr = images[1][:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE]
            ref_hr = images[0][:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE]


            input_lr = input_lr.unsqueeze(0).cuda()
            ref_hr = ref_hr.unsqueeze(0).cuda()

            output = model(ref_hr, input_lr)
            output = output[0].mul_(10000).squeeze()
            # print(h_start_md)
            h_end = h_start + PATCH_SIZE
            w_end = w_start + PATCH_SIZE
            cur_h_start = 0
            cur_h_end = PATCH_SIZE
            cur_w_start = 0
            cur_w_end = PATCH_SIZE

            if i != 0:
                h_start = h_start + PATCH_SIZE // 4
                cur_h_start = PATCH_SIZE // 4

            if i != len(h_index_list) - 1:
                h_end = h_end - PATCH_SIZE // 4
                cur_h_end = cur_h_end - PATCH_SIZE // 4

            if j != 0:
                w_start = w_start + PATCH_SIZE // 4
                cur_w_start = PATCH_SIZE // 4

            if j != len(w_index_list) - 1:
                w_end = w_end - PATCH_SIZE // 4
                cur_w_end = cur_w_end - PATCH_SIZE // 4

            output_image[:, h_start: h_end, w_start: w_end] = \
                output[:, cur_h_start: cur_h_end, cur_w_start: cur_w_end].cpu().detach().numpy()
    zqypackage.write_image(save_dir, image_proj, image_geotrans,output_image,band)


if __name__ == '__main__':
    landsat_dir = r"LC-2019-12-24.tif"
    modis = r"MD-2020-09-20.tif"
    save = r".tif"
    test(Generator(), r".pt", [672, 1024], 256, landsat_dir, modis, save)


E // 4
                cur_h_end = cur_h_end - PATCH_SIZE // 4

            if j != 0:
                w_start = w_start + PATCH_SIZE // 4
                cur_w_start = PATCH_SIZE // 4

            if j != len(w_index_list) - 1:
                w_end = w_end - PATCH_SIZE // 4
                cur_w_end = cur_w_end - PATCH_SIZE // 4

            output_image[:, h_start: h_end, w_start: w_end] = \
                output[:, cur_h_start: cur_h_end, cur_w_start: cur_w_end].cpu().detach().numpy()
    zqypackage.write_image(save_dir, image_proj, image_geotrans,output_image)




def main():
    # 设置随机数种子
    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)
    torch.cuda.manual_seed_all(2021)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('--image_size', default=[2720, 3200], type=int, help='the image size (height, width)')
    parser.add_argument('--patch_size', default=256, type=int, help='training images crop size')
    parser.add_argument('--num_epochs', default=60, type=int, help='train epoch number')
    parser.add_argument('--root_dir', default='D:/pythonProject/swinstfm-main/data/LGC', help='Datasets root directory')
    parser.add_argument('--train_dir', default='D:/pythonProject/swinstfm-main/data/LGC_Train', help='Datasets train directory')
    # 地址
    opt = parser.parse_args()
    IMAGE_SIZE = opt.image_size
    PATCH_SIZE = opt.patch_size
    NUM_EPOCHS = opt.num_epochs

    # 加载LGC数据集
    train_dates = []
    test_dates = []
    for dir_name in os.listdir(opt.root_dir):
        cur_day = int(dir_name.split('_')[1])
        if cur_day not in [331, 347, 363]:
            train_dates.append(dir_name)
        else:
            test_dates.append(dir_name)
    print(train_dates)
    print(test_dates)
    # train(opt, train_dates, test_dates, IMAGE_SIZE, PATCH_SIZE)


if __name__ == '__main__':
    landsat_dir = r"LC-2019-12-24.tif"
    modis = r"MD-2020-09-20.tif"
    save = r".tif"
    test(Generator(), r".pt", [672, 1024], 256, landsat_dir, modis, save)


