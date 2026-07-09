from .custom_mvit import get_custom_mvit
from .custom_r3d import get_custom_r3d
from .custom_video_swin_transformer import get_custom_video_swin_transformer
from .custom_vivit import get_custom_vivit
from .custom_s3d import get_custom_s3d

def get_model(model:int,num_classes:int,phase:int):

    if model == 1:
        return get_custom_mvit(num_classes,phase)
    elif model == 2:
        return get_custom_s3d(num_classes,phase)
    elif model == 3:
        return get_custom_r3d(num_classes,phase)
    elif model == 4:
        return get_custom_video_swin_transformer(num_classes,phase)
    else:
        raise ValueError("Invalid model choice. Please select a number between 1 and 4.")
