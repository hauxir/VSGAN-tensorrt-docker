import sys
sys.path.append("/workspace/tensorrt/")
import vapoursynth as vs
core = vs.core
vs_api_below4 = vs.__api_version__.api_major < 4
core.num_threads = 4
core.std.LoadPlugin(path="/usr/lib/x86_64-linux-gnu/libffms2.so")

from src.cugan import cugan_inference
from src.egvsr import egvsr_inference
from src.pan import PAN_inference

from src.esrgan import ESRGAN_inference



from src.upscale_inference import upscale_inference

def inference_clip(video_path):
    clip = core.ffms2.Source(source=video_path, cache=False)
    clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")

    upscale_model_inference = cugan_inference(fp16=True,scale=2,kind_model="no_denoise")
    #upscale_model_inference = egvsr_inference(scale=4)
    #upscale_model_inference = PAN_inference(scale = 2, fp16 = True)
    #upscale_model_inference = ESRGAN_inference(model_path="/workspace/tensorrt/models/RealESRNet_x4plus.pth", fp16=False, tta=False, tta_mode=1)

    clip = upscale_inference(upscale_model_inference, clip, tile_x=512, tile_y=512, tile_pad=10, pre_pad=0)

    clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")

    return clip
