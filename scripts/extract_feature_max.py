import torch
import torch.nn as nn
import torchvision
from scipy.misc import imread
from torchvision import transforms
from torch.autograd import Variable

import json
import os
import numpy as np
import argparse
import glob

import myfunc as mf


class MyResnet(nn.Module):
    def __init__(self, resnet):
        super(MyResnet, self).__init__()
        self.resnet = resnet

    def forward(self, img, stype):
        x = img.unsqueeze(0)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        if stype == 'mean':
            feature = x.mean(3).mean(2).squeeze()
        else:
            feature = x.max(3)[0].max(2)[0].squeeze()
        return feature


def main(params):
    preprocess = transforms.Compose([
        # trn.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    save_dir = os.path.join(params['out_dir'], params['feature_type'])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for split in ['train', 'val', 'test']:
        feature_file = os.path.join(save_dir, split + "_id_to_feature" + ".npz")

        mf.printn("started to extract features from %s ..." % params['img_dir'])

        model = torchvision.models.resnet152(pretrained=True)
        resnet = MyResnet(model)
        resnet.cuda()
        resnet.eval()
        mf.printn("resnet loaded.")

        id_to_feature = {}
        types = ['.jpg', '.gif', '.png']
        files = []

        for itype in types:
            files.extend(glob.glob("{}/{}/*{}".format(params['img_dir'], split, itype)))
        mf.printn("Found %d image files." % len(files))

        for imfile in files:
            basename = os.path.basename(imfile)
            img_id = basename.split('.')[0]

            try:
                image = imread(imfile)
            except:
                image = np.zeros((224, 224, 3), 'float32')

            if len(image.shape) == 2:
                image = image[:, :, np.newaxis]
                image = np.concatenate((image, image, image), axis=2)

            if image.shape[2] != 3:
                print("Pass %s %s" % (split, basename))
                image = np.zeros((224, 224, 3), 'float32')

            image = image.astype('float32') / 255.0
            image = torch.from_numpy(image.transpose([2, 0, 1]))
            image = Variable(preprocess(image), volatile=True).cuda()
            feature = resnet(image, params['feature_type'])
            mf.printn("Processed %s image:%s" % (split, basename))
            id_to_feature[img_id] = feature.data.cpu().float().numpy()

        np.savez_compressed(feature_file, dict=id_to_feature)
        mf.printn("extracted %d features, saved to %s" % (len(id_to_feature), feature_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='/home/lijiacheng/vist/images/')
    parser.add_argument('--out_dir', type=str, default='/home/lijiacheng/AREL/VIST/resnet_features/')
    parser.add_argument('--feature_type', type=str, default='max', help='max, mean')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)
