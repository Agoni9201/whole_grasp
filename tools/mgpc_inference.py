"""Official MGPC inference loader, confined to the dedicated worker process."""
import argparse
import importlib
import sys

import numpy as np


def load_model(repository, checkpoint_path):
    print('[mgpc-worker] importing torch/encoders',flush=True)
    import torch
    torch.set_num_threads(4)
    import clip
    from transformers import AutoModel, Dinov2Config, Dinov2Model
    from unittest.mock import patch

    if not torch.cuda.is_available():
        raise RuntimeError('MGPC requires an accessible CUDA GPU')
    sys.path.insert(0, str(repository.resolve()))
    sys.path.insert(0, str(repository.resolve()/'extensions/pointnet2_ops_lib'))
    sys.modules['pointnet2_ops._ext'] = importlib.import_module('pointnet2._ext')
    print('[mgpc-worker] loading checkpoint (CPU mmap)',flush=True)
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(checkpoint_path,map_location='cpu',weights_only=True,mmap=True)
    state=checkpoint.get('state_dict',checkpoint)
    state={key.removeprefix('module.').removeprefix('model.'):value for key,value in state.items()}
    prefix='text_encoder.model.clip_model.'
    clip_state={key[len(prefix):]:value for key,value in state.items() if key.startswith(prefix)}
    if not clip_state: raise ValueError('Official MGPC checkpoint lacks CLIP weights')
    key='rgb_encoder.model.embeddings.position_embeddings'
    if key not in state: raise ValueError('Official MGPC checkpoint lacks DINOv2 weights')
    config=Dinov2Config()
    config.image_size=int(round(np.sqrt(state[key].shape[1]-1)))*config.patch_size
    module=importlib.import_module('models.MGPC')
    print('[mgpc-worker] building official model',flush=True)
    def build_dino(*a,**k):
        print('[mgpc-worker] initializing DINOv2 architecture',flush=True)
        result=Dinov2Model(config)
        print('[mgpc-worker] DINOv2 architecture ready',flush=True)
        return result
    def build_clip(*a,**k):
        print('[mgpc-worker] initializing CLIP architecture',flush=True)
        result=clip.model.build_model(clip_state)
        print('[mgpc-worker] CLIP architecture ready',flush=True)
        return result,None
    with patch.object(AutoModel,'from_pretrained',side_effect=build_dino), \
         patch.object(clip,'load',side_effect=build_clip):
        model=module.MGPC()
    print('[mgpc-worker] copying checkpoint weights',flush=True)
    model.load_state_dict(state,strict=True)
    print('[mgpc-worker] strict weights loaded; moving to CUDA',flush=True)
    return model.eval().cuda()


def predict(model, points, image_path, category, seed):
    import torch
    from PIL import Image
    from torchvision import transforms
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    image_transform=transforms.Compose([transforms.Resize(224,interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),transforms.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])])
    image=image_transform(Image.open(image_path).convert('RGB')).unsqueeze(0).cuda()
    tensor=torch.from_numpy(np.asarray(points,np.float32)).unsqueeze(0).cuda()
    with torch.inference_mode():
        prediction,_,_=model.predict(tensor,image,[f'This is a/an "{category}"'])
    result=prediction[0].cpu().numpy()
    if result.shape!=(8192,3) or not np.isfinite(result).all():
        raise ValueError('Invalid MGPC prediction; expected 8192 finite XYZ points')
    return result
