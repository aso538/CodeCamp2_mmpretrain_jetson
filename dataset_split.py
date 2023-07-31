import os
import os.path as osp
import random


def is_pic(filename):
    """ 判断文件是否为图片格式

    Args:
        filename: 文件路径
    """
    suffixes = {'JPEG', 'jpeg', 'JPG', 'jpg', 'BMP', 'bmp', 'PNG', 'png'}
    suffix = filename.strip().split('.')[-1]
    if suffix not in suffixes:
        return False
    return True


def list_files(dirname):
    """ 列出目录下所有文件（包括所属的一级子目录下文件）

    Args:
        dirname: 目录路径
    """

    def filter_file(f):
        if f.startswith('.'):
            return True
        return False

    all_files = list()
    dirs = list()
    for f in os.listdir(dirname):
        if filter_file(f):
            continue
        if osp.isdir(osp.join(dirname, f)):
            dirs.append(f)
        else:
            all_files.append(f)
    for d in dirs:
        for f in os.listdir(osp.join(dirname, d)):
            if filter_file(f):
                continue
            if osp.isdir(osp.join(dirname, d, f)):
                continue
            all_files.append(osp.join(d, f))
    return all_files


def split_imagenet_dataset(dataset_dir, val_percent, test_percent, save_dir):
    all_files = list_files(dataset_dir)
    label_list = list()
    train_image_anno_list = list()
    val_image_anno_list = list()
    test_image_anno_list = list()
    for file in all_files:
        if not is_pic(file):
            continue
        label, image_name = osp.split(file)
        if label not in label_list:
            label_list.append(label)
    label_list = sorted(label_list)

    for i in range(len(label_list)):
        image_list = list_files(osp.join(dataset_dir, label_list[i]))
        image_anno_list = list()
        for img in image_list:
            image_anno_list.append([osp.join(label_list[i], img), i])
        random.shuffle(image_anno_list)
        image_num = len(image_anno_list)
        val_num = int(image_num * val_percent)
        test_num = int(image_num * test_percent)
        train_num = image_num - val_num - test_num

        train_image_anno_list += image_anno_list[:train_num]
        val_image_anno_list += image_anno_list[train_num:train_num + val_num]
        test_image_anno_list += image_anno_list[train_num + val_num:]

    with open(
            osp.join(save_dir, 'train_list.txt'), mode='w',
            encoding='utf-8') as f:
        for x in train_image_anno_list:
            file, label = x
            f.write('{} {}\n'.format(file, label))
    with open(
            osp.join(save_dir, 'val_list.txt'), mode='w',
            encoding='utf-8') as f:
        for x in val_image_anno_list:
            file, label = x
            f.write('{} {}\n'.format(file, label))
    if len(test_image_anno_list):
        with open(
                osp.join(save_dir, 'test_list.txt'), mode='w',
                encoding='utf-8') as f:
            for x in test_image_anno_list:
                file, label = x
                f.write('{} {}\n'.format(file, label))
    # 创建label标签
    with open(
            osp.join(save_dir, 'labels.txt'), mode='w', encoding='utf-8') as f:
        for l in sorted(label_list):
            f.write('{}\n'.format(l))

    return len(train_image_anno_list), len(val_image_anno_list), len(
        test_image_anno_list)


def test_cls_split(input_dir=r'./pokemon',
                   val_percent=0.2,
                   test_percent=0.1,
                   ):
    '''
        val_num = int(image_num * val_percent)
        test_num = int(image_num * test_percent)
        train_num = image_num - val_num - test_num
    '''

    split_imagenet_dataset(
        dataset_dir=input_dir,
        val_percent=val_percent,
        test_percent=test_percent,
        save_dir=input_dir

    )


if __name__ == '__main__':
    # input_dir = './datasets/cls_2'
    test_cls_split()
