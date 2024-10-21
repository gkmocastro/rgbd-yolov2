import copy
from PIL import Image
import torch
from torchvision import transforms as tf
import brambox.boxes as bbb
import brambox.transforms as bbtf
import lightnet as ln

__all__ = ['EPFLData']

def rgb_id(img_id):
    return f'data/rgb/{img_id}.png'

def depth_id(img_id):
    return f'data/depth/{img_id}.png'

class EPFLData(ln.models.BramboxDataset):
    def __init__(self, anno, params, augment=True):
        super().__init__('anno_pickle', anno, params.input_dimension, params.class_label_map)
        self.channels = params.channels
        if self.channels not in ('RGB', 'D', 'RGBD'):
            raise ValueError(f'Channels should be one of RGB,D or RGBD [{self.channels}]')

        # Augmentation
        lb = ln.data.transform.Letterbox(dataset=self)
        lb.fill_color = 0
        self.rgb_tf = lambda img: Image.merge('RGB', img.split()[:3])
        self.img_tf = ln.data.transform.Compose([lb, tf.ToTensor()])
        self.anno_tf = ln.data.transform.Compose([lb])
        if augment:
            rf  = ln.data.transform.RandomFlip(params.flip)
            rc  = ln.data.transform.RandomJitter(params.jitter, False, (0.5, 0.5), fill_color=0)
            self.rgb_tf = ln.data.transform.RandomHSV(params.hue, params.saturation, params.value)
            self.img_tf[0:0] = [rc, rf]
            self.anno_tf[0:0] = [rc, rf]

        # Channel Mixing
        self.mixer = bbtf.ChannelMixer(4)
        self.mixer.set_channels([(0,0), (0,1), (0,2), (1,0)])

    @ln.models.BramboxDataset.resize_getitem
    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError(f'list index out of range [{index}/{len(self)-1}]')

        # Load
        if self.channels in ('RGB', 'RGBD'):
            img = self.rgb_tf(Image.open(rgb_id(self.keys[index])))
        else:
            img = Image.open(depth_id(self.keys[index]))
        if self.channels == 'RGBD':
            d = Image.open(depth_id(self.keys[index]))
            img = self.mixer(img, d)
        anno = copy.deepcopy(self.annos[self.keys[index]])

        # Transform
        img = self.img_tf(img)
        anno = self.anno_tf(anno)

        for a in anno:
            if a.lost or a.truncated_fraction >= 1:
                a.ignore = True

        return img, anno
