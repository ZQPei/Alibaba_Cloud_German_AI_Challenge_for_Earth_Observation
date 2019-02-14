# -*- coding: utf-8 -*-
'''
 * @Author: ZQ.Pei 
 * @Date: 2018-11-24 23:10:02 
 * @Last Modified by:   ZQ.Pei 
 * @Last Modified time: 2018-11-24 23:10:02 
'''

import torch
import torch.nn.functional as F

import numbers
import random

def pad(img, padding):
    '''
    input:
        img: torch.Tensor type image
            [C x H x W]
    return:
        padded image
    '''
    return F.pad(img, (padding, padding, padding, padding))

class H5RandomCrop(object):
    """Crop the given H5 Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (H5 Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        _, h, w = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (H5 Image): Image to be cropped.
            [18x32x32]
        Returns:
            H5 Image: Cropped image.
        """
        if self.padding > 0:
            img = pad(img, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            # img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
            pass
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            # img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))
            pass

        i, j, h, w = self.get_params(img, self.size)
        
        return img[:,i:i+h,j:j+w]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class H5RandomHorizontalFlip(object):
    """Horizontally flip the given H5 Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (H5 Image): Image to be flipped.

        Returns:
            H5 Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return img.index_select(2,torch.arange(31,-1,-1).long())
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class H5HorizontalFlip(object):
    """Horizontally flip the given H5 Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """


    def __call__(self, img):
        """
        Args:
            img (H5 Image): Image to be flipped.

        Returns:
            H5 Image: Randomly flipped image.
        """        
        return img.index_select(2,torch.arange(31,-1,-1).long())

class H5RandomVerticalFlip(object):
    """Vertically flip the given H5 Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (H5 Image): Image to be flipped.

        Returns:
            H5 Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return img.index_select(1,torch.arange(31,-1,-1).long())
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
        


class H5VerticalFlip(object):
    """Vertically flip the given H5 Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """



    def __call__(self, img):
        """
        Args:
            img (H5 Image): Image to be flipped.

        Returns:
            H5 Image: Randomly flipped image.
        """

        return img.index_select(1,torch.arange(31,-1,-1).long())

    # def __repr__(self):
    #     return self.__class__.__name__ + '(p={})'.format(self.p)


class H5RandomRotate(object):
    """Rotate the given H5 Image randomly with a fixed probability.

    """
    def __init__(self):
        self.angles = (0, 90, 180, 270)

    def __call__(self, img):
        """
        Args:
            img (H5 Image): Image to be flipped.

        Returns:
            H5 Image: Randomly rotated image.
        """
        rotate_degree = random.choice(self.angles)
        T  = False if rotate_degree == 180 or rotate_degree == 0 else True
        VF = False if rotate_degree == 270 or rotate_degree == 0 else True
        HF = False if rotate_degree ==  90 or rotate_degree == 0 else True
        if T:
            img = img.transpose(1,2)
        if VF:
            img = img.index_select(1, torch.arange(31,-1,-1).long())
        if HF:
            img = img.index_select(2, torch.arange(31,-1,-1).long())
        return img

class H5Rotate(object):
    """Rotate the given H5 Image randomly with a fixed probability.

    """
    # def __init__(self):
    #     # self.angles = (0, 90, 180, 270)

    def __call__(self, img, rotate_degree):
        """
        Args:
            img (H5 Image): Image to be flipped.

        Returns:
            H5 Image: Randomly rotated image.
        """
        # rotate_degree = random.choice(self.angles)
        T  = False if rotate_degree == 180 or rotate_degree == 0 else True
        VF = False if rotate_degree == 270 or rotate_degree == 0 else True
        HF = False if rotate_degree ==  90 or rotate_degree == 0 else True
        if T:
            img = img.transpose(1,2)
        if VF:
            img = img.index_select(1, torch.arange(31,-1,-1).long())
        if HF:
            img = img.index_select(2, torch.arange(31,-1,-1).long())
        return img


if __name__ == "__main__":
    x = torch.randn((32,32,18))
    x = x.permute(2,0,1)
    randomcrop = H5RandomCrop(32,4)
    randomHflip = H5RandomHorizontalFlip()
    randomVflip = H5RandomVerticalFlip()
    randomRotate = H5RandomRotate()
    import ipdb; ipdb.set_trace()
    y = randomcrop(x)
    y = randomHflip(y)
    y = randomVflip(y)
    y = randomRotate(y)
