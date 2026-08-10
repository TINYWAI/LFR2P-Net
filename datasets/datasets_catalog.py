# Required dataset entry keys
_PREFIX = 'image_directory'
_SOURCE_INDEX = 'image_and_label_list_file'
_MEAN = 'rgb_mean'
_STD = 'rgb_std'
_CLASSES_LIST = 'class_names_list'
_COLORS = 'classes_colors'

# Available datasets
_DATASETS = {
    'BRIGHT': {
        _PREFIX: '/data3/wty/data/BRIGHT',
        _SOURCE_INDEX: {
            'train': 'datasets/bright/train_set.txt',
            'val': 'datasets/bright/val_set.txt',
            'test': 'datasets/bright/test_set.txt'
        },
        _MEAN: [123.675, 116.28, 103.53],
        _STD: [58.395, 57.12, 57.375],
        _COLORS: {
            'Background': [255, 255, 255],
            'Intact': [108, 178, 125],
            'Damaged': [219, 190, 144],
            'Destroyed': [163, 78, 73],
        }
    },
}


def datasets():
    """Retrieve the list of available dataset names."""
    return _DATASETS.keys()


def contains(name):
    return name in _DATASETS.keys()


def get_prefix(name):
    return _DATASETS[name][_PREFIX]


def get_source_index(name):
    return _DATASETS[name][_SOURCE_INDEX]


def get_da_classes(name):
    return len(_DATASETS[name][_COLORS])


def get_mean(name):
    return _DATASETS[name][_MEAN]


def get_std(name):
    return _DATASETS[name][_STD]


def get_names_list(name):
    return _DATASETS[name][_CLASSES_LIST]


def get_vis_colors(name):
    return _DATASETS[name][_COLORS]
