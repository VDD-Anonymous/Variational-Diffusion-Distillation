from common.models.vision_encoders.model_getter import get_resnet
from common.models.vision_encoders.multi_image_obs_encoder import MultiImageObsEncoder

def get_visual_encoder(params):
    rgb_model_params = params["rgb_model_params"]
    rgb_model = get_resnet(rgb_model_params['input_shape'], rgb_model_params['output_size'])
    return MultiImageObsEncoder(rgb_model=rgb_model, **params)
