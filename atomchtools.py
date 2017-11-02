'''
Functions:
    ApplyCredits
    ApplyImageMask
    Tp7DebandMask
    JensenLineMask
    ApplyF3kdbGrain
    MakeTestEncode
    DiffCreditlessMask
    DiffRescaleMask
    ApplyMaskOnLuma
'''

import vapoursynth as vs
import mvsfunc as mvf
import havsfunc as haf
import cooldegrain
import descale as dsc

def ApplyCredits(credits, nc, fixed_nc, LumaOnly=True):
    core = vs.get_core()
    funcName = 'ApplyCredits'

    if not isinstance(credits, vs.VideoNode):
        raise TypeError(funcName + ': \"credits\" must be a clip!')
    if not isinstance(nc, vs.VideoNode):
        raise TypeError(funcName + ': \"nc\" must be a clip!')
    if not isinstance(fixed_nc, vs.VideoNode):
        raise TypeError(funcName + ': \"fixed_nc\" must be a clip!')

    if LumaOnly is True:
        credits_ = mvf.GetPlane(credits, 0)
        nc = mvf.GetPlane(nc, 0)
        fixed_nc = mvf.GetPlane(fixed_nc, 0)
    else:
        credits_ = credits

    averaged = core.misc.AverageFrames([credits_, nc, fixed_nc], [1, -1, 1], scale=1)
    if LumaOnly is True:
        averaged = core.std.ShufflePlanes([averaged, mvf.GetPlane(credits, 1), mvf.GetPlane(credits, 2)], planes=[0, 0, 0], colorfamily=credits.format.color_family)
    return averaged
    
def CopyColors(clip, colors):
    core = vs.get_core()
    funcName = 'CopyColors'

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" must be a clip!')
    if not isinstance(colors, vs.VideoNode):
        raise TypeError(funcName + ': \"colors\" must be a clip!')

    final = core.std.ShufflePlanes([mvf.GetPlane(clip, 0), mvf.GetPlane(colors, 1), mvf.GetPlane(colors, 2)], planes=[0, 0, 0], colorfamily=colors.format.color_family)
    return final

def ApplyImageMask(source, replacement, immask, LumaOnly=True):
    core = vs.get_core()
    filemask = core.imwrif.Read(immask).resize.Point(format=source.format.id, matrix_s="709", chromaloc_s="top_left")

    NumPlanes = source.format.num_planes
    if LumaOnly is True or NumPlanes == 1:
        planes = [0]
        filemask = mvf.GetPlane(filemask, 0)
        source_ = mvf.GetPlane(source, 0)
        replacement_ = mvf.GetPlane(replacement, 0)
    else:
        planes = [0,1,2]
        source_ = source
        replacement_ = replacement
    
    mask = core.std.Binarize(filemask, 128).std.Maximum().std.Deflate()
    masked = core.std.MaskedMerge(source_, replacement_, mask, planes)
    if LumaOnly is True and NumPlanes > 1:
        masked = core.std.ShufflePlanes([masked, mvf.GetPlane(source, 1), mvf.GetPlane(source, 2)], planes=[0, 0, 0], colorfamily=source.format.color_family)
    return masked

def Tp7DebandMask(clip, thr=10, scale=1, rg=True):
    core = vs.get_core()
    funcName = 'Tp7DebandMask'

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" must be a clip!')

    mask = core.std.Prewitt(clip, [0,1,2], scale)
    mask = core.std.Expr(mask, ['x {mthr} < 0 255 ?'.format(mthr=thr)])

    if rg is True:
        mask = core.rgvs.RemoveGrain(mask, 3).rgvs.RemoveGrain(4)

    mask_uv = core.std.Expr([mvf.GetPlane(mask, 1), mvf.GetPlane(mask, 2)], ['x y +']).resize.Point(mask.width, mask.height)
    mask_yuv_on_y = core.std.Expr([mvf.GetPlane(mask, 0), mask_uv], ['x y +']).std.Maximum()
    return mask_yuv_on_y

def JensenLineMask(clip, thr_y=7, thr_u=8, thr_v=8, scale=1, rg=True):
    core = vs.get_core()
    funcName = 'JensenLineMask'

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" must be a clip!')

    clip_y = mvf.GetPlane(clip, 0)
    clip_u = mvf.GetPlane(clip, 1)
    clip_v = mvf.GetPlane(clip, 2)

    mask_y = core.std.Prewitt(clip_y, [0], scale)
    mask_u = core.std.Prewitt(clip_u, [0], scale)
    mask_v = core.std.Prewitt(clip_v, [0], scale)

    if rg is True:
        mask_y = core.rgvs.RemoveGrain(mask_y, 3).rgvs.RemoveGrain(4)
        mask_u = core.rgvs.RemoveGrain(mask_u, 3).rgvs.RemoveGrain(4)
        mask_v = core.rgvs.RemoveGrain(mask_v, 3).rgvs.RemoveGrain(4)

    mask_uv = core.std.Expr([core.std.Expr(mask_u, ['x {mthr} < 0 255 ?'.format(mthr=thr_u)]), core.std.Expr(mask_v, ['x {mthr} < 0 255 ?'.format(mthr=thr_v)])], ['x y +']).resize.Point(mask_y.width, mask_y.height)
    mask_yuv_on_y = core.std.Expr([core.std.Expr(mask_y, ['x {mthr} < 0 255 ?'.format(mthr=thr_y)]), mask_uv], ['x y +']).std.Maximum().std.Deflate()
    return mask_yuv_on_y
    
    
def ApplyF3kdbGrain(clip, dfttest_sigma=25, dfttest_tbsize=3, thsad=80, thsadc=80, detect_val=80, grain_val=120, dyn_grain=False, tv_range=True):
    core = vs.get_core()
    funcName = 'ApplyF3kdbGrain'

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" must be a clip!')

    clip16 = core.fmtc.bitdepth(clip, bits=16)

    repairmask = JensenLineMask(clip, 8, 12, 12).std.Deflate().fmtc.bitdepth(bits=16)

    pf = core.dfttest.DFTTest(clip16, sigma=dfttest_sigma, tbsize=dfttest_tbsize, planes=[0])

    filtered = cooldegrain.CoolDegrain(clip, tr=1, thsad=100, thsadc=100, bits=16, blksize=8, overlap=4, pf=pf)
    filtered = core.f3kdb.Deband(filtered, dither_algo=3, y=detect_val, cb=detect_val, cr=detect_val, grainy=grain_val, grainc=grain_val, dynamic_grain=dyn_grain, keep_tv_range=tv_range, output_depth=16)
    filtered = core.std.MaskedMerge(filtered, clip16, repairmask, planes=[0,1,2], first_plane=True)
    return filtered

def MergeHalf(source, aaclip, right=True):
    core = vs.get_core()
    funcName = 'MergeHalf'

    if not isinstance(source, vs.VideoNode):
        raise TypeError(funcName + ': \"source\" must be a clip!')
    if not isinstance(aaclip, vs.VideoNode):
        raise TypeError(funcName + ': \"aaclip\" must be a clip!')
    if right is True:
        source_part = core.std.CropRel(source, right=source.width//2)
        aaclip_part = core.std.CropRel(aaclip, left=source.width//2)
        merged = core.std.StackHorizontal([source_part, aaclip_part])
    else:
        source_part = core.std.CropRel(source, left=source.width//2)
        aaclip_part = core.std.CropRel(aaclip, right=source.width//2)
        merged = core.std.StackHorizontal([aaclip_part, source_part])
    return merged

def MakeTestEncode(clip):
    core = vs.get_core()
    funcName = 'MakeTestEncode'

    t1=clip.num_frames/100
    t2=int(t1*2)
    clip = core.std.SelectEvery(clip, cycle=t2, offsets=range(50))
    clip = core.std.AssumeFPS(clip, fpsnum=24000, fpsden=1001)

    return clip

def DiffCreditlessMask(titles, nc):
    core = vs.get_core()
    funcName = 'DiffCreditlessMask'
    
    test = core.std.MakeDiff(titles, nc, [0])
    test = mvf.GetPlane(test, 0)
    test = test.std.Prewitt().std.Expr('x 25 < 0 x ?').std.Expr('x 2 *')
    test = core.rgvs.RemoveGrain(test, 4).std.Expr('x 30 > 255 x ?')
    return test

def DiffRescaleMask(clip, descale_h=720, kernel='bicubic', mthr=55):
    core = vs.get_core()
    funcName = 'DiffRescaleMask'

    descale_w = haf.m4((clip.width * descale_h) / clip.height)
    dclip = dsc.Descale(clip, descale_w, descale_h, kernel=kernel)
    uclip = core.fmtc.resample(dclip, clip.width, clip.height, kernel=kernel).fmtc.bitdepth(bits=8)
    uclip = mvf.GetPlane(uclip, 0)
    clip = mvf.GetPlane(clip, 0)
    diff = core.std.MakeDiff(clip, uclip)
    mask = diff.rgvs.RemoveGrain(2).rgvs.RemoveGrain(2).hist.Luma()
    mask = mask.std.Expr('x {thr} < 0 x ?'.format(thr=mthr))
    mask = mask.std.Prewitt().std.Maximum().std.Maximum().std.Deflate()
    return mask

def ApplyMaskOnLuma(src, aa, mask):
    core = vs.get_core()
    funcName = 'ApplyMaskOnLuma'

    src_y = mvf.GetPlane(src, 0)
    aa_y = mvf.GetPlane(aa, 0)
    mask_y = mvf.GetPlane(mask, 0)
    masked = core.std.MaskedMerge(aa_y, src_y, mask_y)
    result = core.std.ShufflePlanes([masked, mvf.GetPlane(src, 1), mvf.GetPlane(src, 2)], planes=[0, 0, 0], colorfamily=src.format.color_family)
    return result
