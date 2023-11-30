from Image_Style_Transfer.basic_style_transfer import PixelLevelDA

__method_factory = {

    # Basic Domain Adaptation methods
    'fda': lambda img, images: PixelLevelDA.fda(img, images, beta_limit=0.13),
    'pixel_distribution': lambda img, images: PixelLevelDA.pixel_distribution(img, images, transform_type='pca'),
    'hist_matching': lambda img, images: PixelLevelDA.hist_matching(img, images)
}


def get_da_method(method_type):
    """A function wrapper for building a dim reduction.
    """
    method_types = list(__method_factory.keys())

    if method_type not in method_types:
        raise KeyError(
            'Unknown method: {}. Must be one of {}'.format(method_type, method_types)
        )
    return __method_factory[method_type]
