import vapoursynth as vs
core = vs.core
from vsdeinterlace import QTempGaussMC







source = '/home/lyra/D/TT Project/Testing/2023-08-27_10-56-39.mkv'

clip = core.bs.VideoSource(source=source)

clip = core.resize.Bicubic(clip, format=vs.YUV422P16)

clip = (QTempGaussMC(clip,input_type=1)
    .prefilter() #tr=2
    .analyze() #preset=vsdenoise.MVToolsPreset.HQ_SAD
    .denoise()
    .basic()
    .source_match(mode=QTempGaussMC.SourceMatchMode.TWICE_REFINED) #BASIC,
    .lossless() #mode=QTempGaussMC.LosslessMode.POSTSMOOTH
    .sharpen()
    .back_blend()
    .sharpen_limit()
    .final()
    .motion_blur() #fps_divisor=1
    .deinterlace()
    )

clip = core.resize.Bicubic(clip, format=vs.YUV420P10, dither_type="error_diffusion") #, transfer=1, primaries=1, matrix=1

clip.set_output(index=0)
