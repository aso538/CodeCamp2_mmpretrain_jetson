# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

from mmengine import fileio
from mmengine.logging import MMLogger

from mmcls.registry import DATASETS
from .custom import CustomDataset

GARBAGE_CATEGORIES = ('recyclable_metallic food cans', 'other garbage _PE plastic bags',' other garbage _ soiled plastic ',' other garbage _ wet paper towels', 'other garbage _ Adhesive bandage', 'recyclable _ hair dryer', 'other garbage _ post it notes',' other garbage _ glasses', 'kitchen garbage _ cake', 'recyclable _ backpacks',' recyclable _ cloth strips', 'kitchen garbage _ tofu', 'other garbage _ lunch boxes',' kitchen garbage _ potato chips', 'Hazardous waste _ glass tubes',' recyclable Things_ Ashtray ',' recyclable_ Spoon, other garbage_ Cigarette butts, kitchen waste_ Bingtanghulu ',' Kitchen waste_ Watermelon peel, recyclable_ Stuffed toy ',' recyclables_ Quilt, recyclable_ Cutting board, recyclable_ Express paper bags, recyclable items_ Chair, recyclable_ Paper bags, kitchen waste_ Apples, recyclable_ Keyboard, kitchen waste_ French fries, recyclable_ Tweezers, recyclable_ Table lamp, recyclable_ Electric iron, recyclable_ Table, recyclable_ Glass pot, other garbage_ Scouring pad ',' kitchen waste_ Vermicelli, recyclable_ Cards, kitchen waste_ Fava beans, recyclable_ Cosmetic bottles, other waste_ Straw hat, kitchen waste_ Banana peel ',' recyclable_ Paper, recyclable_ Gas bottle, recyclable_ Clocks and watches, recyclable_ Hot water bottle, recyclable_ Wood spatula, recyclable_ Scissors, kitchen waste_ Big bones, recyclable items_ Electric curling stick, recyclable_ Insulating cup, kitchen waste_ Bread, kitchen waste_ Babao Congee ',' recyclable_ Mouse, kitchen waste_ Chicken wings, recyclable_ Bag, recyclable_ Wooden comb, recyclable_ Ruler, other garbage_ Towels, kitchen waste_ Strawberries, recyclable_ Milk box, recyclable waste_ Hat, recyclable_ Card, kitchen waste_ Tomatoes, recyclable_ Kettle, kitchen waste_ Fish bones, recyclable_ Carton, recyclable_ Router, other garbage_ Masks, recyclable_ Mobile phone ',' Hazardous waste_ Dry cell ',' recyclable_ Pillows, other garbage_ Desiccant, kitchen waste_ Intestine, recyclable_ Gas stove, recyclable_ Earphones, recyclable_ Cake box, recyclable_ Umbrella, other garbage_ Bamboo chopsticks, recyclable_ Shoulder bag, recyclable_ Stapler ',' Recyclable_ Remote control, kitchen waste_ Carrots, recyclable_ Nails, kitchen waste_ Hami melon ',' recyclable_ Cage ',' Hazardous waste_ Light bulb, other waste_ Chicken feather duster, other garbage_ Kitchen gloves, kitchen waste_ Sugarcane, other waste_ Disposable cup, recyclable_ Patch panels, recyclable_ Plastic toys, recyclable items_ Wire ball ',' Hazardous waste_ Expired drugs, recyclable materials_ Power Bank, recyclable_ Tyres', 'Hazardous waste_ Glue, other waste_ Flyswatter ',' Kitchen waste_ Tea, other garbage_ Lighter, recyclable_ Foam box ',' kitchen waste_ Meat, recyclable_ Edible oil barrel, recyclable_ Bracelet, recyclable_ Shampoo bottles, kitchen waste_ Vegetable roots and leaves, recyclable_ Electric fan ',' Hazardous waste_ Insecticides, recyclable_ Fire extinguisher, recyclable_ Table tennis racket ',' Hazardous waste_ Button batteries, kitchen waste_ Coffee ',' Hazardous waste_ Medicine bottles, other garbage_ Pregnancy test rod, recyclable_ Beverage bottles, recyclable items_ Shoes, recyclable items_ Plastic box, recyclable_ Drink can ',' kitchen waste_ Hamburg, recyclable_ Wine bottle, recyclable_ Plug wires, kitchen waste_ Residues, leftovers, and other garbage_ Broken flower pots and bowls, recyclable_ Plastic bowls and basins, recyclable_ Calculator, recyclable_ Scale, recyclable_ Wooden bucket, recyclable material_ Archive bag, other garbage_ Pen, other garbage_ Toothpicks, recyclable_ Plastic hanger, kitchen waste_ Melon seeds, recyclable_ Seasoning bottle, recyclable_ Electric shaver ',' recyclable_ Circuit board, kitchen waste_ Oranges, recyclable_ Pump, kitchen waste_ Egg tarts, kitchen waste_ Roasted chicken, other garbage_ Tape ',' Hazardous waste_ Thermometer, other waste_ Toothbrush, recyclable_ Socks, recyclable_ Pot, kitchen waste_ Pitaya ',' kitchen waste_ Eggs, other garbage_ Bath towel ',' Hazardous waste_ Battery, recyclable_ Trolley box, other garbage_ Disposable cotton swabs, kitchen waste_ Biscuits, other garbage_ Alteration tape, kitchen waste_ Pineapple, kitchen waste_ Badan wood, recyclable_ Glass cup, recyclable_ Induction cooking ',' recyclable_ Old clothes, kitchen waste_ Ice cream')

@DATASETS.register_module()
class Garbage(CustomDataset):
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    METAINFO = {'classes': GARBAGE_CATEGORIES}

    def __init__(self,
                 data_root: str = '',
                 split: str = '',
                 data_prefix: Union[str, dict] = '',
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 **kwargs):
        kwargs = {'extensions': self.IMG_EXTENSIONS, **kwargs}

        if split:
            splits = ['train', 'val', 'test']
            assert split in splits, \
                f"The split must be one of {splits}, but get '{split}'"

            if split == 'test':
                logger = MMLogger.get_current_instance()
                logger.info(
                    'Since the ImageNet1k test set does not provide label'
                    'annotations, `with_label` is set to False')
                kwargs['with_label'] = False

            data_prefix = split if data_prefix == '' else data_prefix

            if ann_file == '':
                _ann_path = fileio.join_path(data_root, 'meta', f'{split}.txt')
                if fileio.exists(_ann_path):
                    ann_file = fileio.join_path('meta', f'{split}.txt')

        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file=ann_file,
            metainfo=metainfo,
            **kwargs)

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Root of dataset: \t{self.data_root}',
        ]
        return body
