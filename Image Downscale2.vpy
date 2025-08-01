import vapoursynth as vs
core = vs.core
#from vsscale import ssim_downsample







source = '/home/lyra/D/TT Project/Testing/2023-08-27_10-56-39.mkv'

clip = core.lsmas.LWLibavSource(source=source)

clip = core.fmtc.resample(clip=clip, kernel="lanczos", w=1280, h=720, taps=2, fh=1.047124406, fv=1.047124406)

clip.set_output(0)
