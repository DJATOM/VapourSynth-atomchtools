from vapoursynth import core, VideoNode, YUV, GRAY # pylint: disable=no-name-in-module
import collections
import cooldegrain
import descale as dsc
import havsfunc as haf
import inspect
from typing import Union

try:
    from collections.abc import Sequence
except AttributeError:
    from collections import Sequence

__version__ = 0.86

'''
Atomch Tools
Version 0.86 from 09.04.2020

Functions:
    ApplyCredits
    CopyColors
    ApplyImageMask
    Tp7DebandMask
    JensenLineMask
    ApplyF3kdbGrain
    MergeHalf
    MakeTestEncode
    DiffCreditlessMask
    DiffRescaleMask
    DiffOn2FramesMask
    ApplyMaskOnLuma
    eedi3Scale
    RfsMany
    rfs
    retinex_edgemask
    kirsch
    m4
'''

def ApplyCredits(credits: VideoNode, nc: VideoNode, fixed_nc: VideoNode, luma_only: bool = True) -> VideoNode:
    ''' Convenient helper for applying credits on processed creditless videos (OP/ED) '''
    funcName = 'ApplyCredits'
    if not isinstance(credits, VideoNode):
        raise TypeError(f'{funcName}: "credits" must be a clip!')
    if not isinstance(nc, VideoNode):
        raise TypeError(f'{funcName}: "nc" must be a clip!')
    if not isinstance(fixed_nc, VideoNode):
        raise TypeError(f'{funcName}: "fixed_nc" must be a clip!')
    assert credits.num_frames == nc.num_frames == fixed_nc.num_frames, TypeError(f'{funcName}: input clips are not even! credits: {credits.num_frames}, nc - {nc.num_frames}, fixed_nc - {fixed_nc.num_frames}!')
    if luma_only is True:
        credits_ = core.std.ShufflePlanes(credits, 0, GRAY)
        nc = core.std.ShufflePlanes(nc, 0, GRAY)
        fixed_nc = core.std.ShufflePlanes(fixed_nc, 0, GRAY)
    else:
        credits_ = credits
    averaged = core.std.Expr([credits_, nc, fixed_nc], ['x y - z +'])
    if luma_only is True:
        averaged = core.std.ShufflePlanes([averaged, credits], planes=[0, 1, 2], colorfamily=credits.format.color_family)
    return averaged

def CopyColors(clip: VideoNode, colors: VideoNode) -> VideoNode:
    ''' Applies colors components from one clip to another '''
    funcName = 'CopyColors'
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{funcName}: "clip" must be a clip!')
    if not isinstance(colors, VideoNode):
        raise TypeError(f'{funcName}: "colors" must be a clip!')
    assert clip.num_frames == colors.num_frames, TypeError(f'{funcName}: input clips are not even! clip: {clip.num_frames}, colors: {colors.num_frames}!')
    return core.std.ShufflePlanes([clip, colors], planes=[0, 1, 2], colorfamily=colors.format.color_family)

def ApplyImageMask(source: VideoNode, replacement: VideoNode, image_mask: str = None, luma_only: bool = True, binarize_threshold: int = 17, scaled_binarize: bool = False, preview: bool = False, first_plane_mask: bool = True, blur: bool = False) -> VideoNode:
    ''' Applies custom (hand-drawn) image as static mask for two clips '''
    funcName = 'ApplyImageMask'
    if not isinstance(source, VideoNode):
        raise TypeError(f'{funcName}: "source" must be a clip!')
    if not isinstance(replacement, VideoNode):
        raise TypeError(f'{funcName}: "replacement" must be a clip!')
    if not isinstance(image_mask, VideoNode):
        MaskReader = core.imwrif.Read if hasattr(core, 'imwrif') else core.imwri.Read
        filemask = MaskReader(image_mask).resize.Point(format=source.format.id, matrix_s="709", chromaloc_s="top_left")
    elif isinstance(image_mask, VideoNode):
        filemask = image_mask
    else:
        raise TypeError(f'{funcName}: "image_mask" has unsupported type!')
    mum_planes = source.format.num_planes
    if luma_only is True or mum_planes == 1:
        planes = [0]
        filemask = core.std.ShufflePlanes(filemask, 0, GRAY)
        source_ = core.std.ShufflePlanes(source, 0, GRAY)
        replacement_ = core.std.ShufflePlanes(replacement, 0, GRAY)
    else:
        planes = [0,1,2]
        source_ = source
        replacement_ = replacement
    assert source.num_frames == replacement.num_frames, TypeError(f'{funcName}: input clips are not even! source: {source.num_frames}, replacement: {replacement.num_frames}!')
    if not scaled_binarize:
        max_pixel_value = (256 << (source.format.bits_per_sample - 8)) - 1
        binarize_threshold = min(binarize_threshold << (source.format.bits_per_sample - 8), max_pixel_value)
    filemask = core.std.Expr(filemask, f'x {binarize_threshold} < 0 x ?').std.Maximum().std.Deflate()
    if preview:
        replacement_ = core.std.Merge(filemask, replacement_, 0.5)
    masked = core.std.MaskedMerge(source_, replacement_, filemask, planes, first_plane_mask)
    if blur:
        filemask = filemask.std.Maximum().std.Inflate()
        masked_blurry = haf.MinBlur(masked, 3)
        masked = core.std.MaskedMerge(masked, masked_blurry, filemask, planes, first_plane_mask)
    if luma_only is True and mum_planes > 1:
        masked = core.std.ShufflePlanes([masked, source], planes=[0, 1, 2], colorfamily=source.format.color_family)
    return masked

def Tp7DebandMask(clip: VideoNode, thr: int = 10, scale: int = 1, rg: bool = True) -> VideoNode:
    ''' Ported Tp7's mask for detecting chroma lines '''
    funcName = 'Tp7DebandMask'
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{funcName}: "clip" must be a clip!')
    bits = clip.format.bits_per_sample
    maxvalue = (1 << bits) - 1
    thr = thr * maxvalue // 0xFF
    mask = core.std.Prewitt(clip, [0,1,2], scale)
    mask = core.std.Expr(mask, [f'x {thr} < 0 {maxvalue} ?'])
    if rg is True:
        mask = core.rgvs.RemoveGrain(mask, 3).rgvs.RemoveGrain(4)
    mask_uv = core.std.Expr([mask.std.ShufflePlanes(1, GRAY), mask.std.ShufflePlanes(2, GRAY)], ['x y +']).resize.Point(mask.width, mask.height)
    mask_yuv_on_y = core.std.Expr([core.std.ShufflePlanes(mask, 0, GRAY), mask_uv], ['x y +']).std.Maximum()
    return mask_yuv_on_y

def JensenLineMask(clip: VideoNode, thr: Union[int, Sequence] = (7, 8, 8), scale: int = 1, rg: bool = True) -> VideoNode:
    ''' A modified one to Jensen's needs '''
    funcName = 'JensenLineMask'
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{funcName}: "clip" must be a clip!')
    if isinstance(thr, int):
        thr_y, thr_u, thr_v = [thr] * 3
    elif isinstance(thr, Sequence):
        if len(thr) == 2:
            thr_y, [thr_u, thr_v] = thr[0], [thr[1]] * 2
        elif len(thr) == 3:
            thr_y, thr_u, thr_v = thr
        else:
            raise ValueError(f'{funcName}: "thr" in Sequence mode must have 2 or 3 values at most!')
    else:
        raise ValueError(f'{funcName}: "thr" got wrong set of values!')
    mask = core.std.Prewitt(clip, [0,1,2], scale)
    if rg is True:
        mask = mask.rgvs.RemoveGrain(3).rgvs.RemoveGrain(4)
    bits = clip.format.bits_per_sample
    maxvalue = (1 << bits) - 1
    thr_y = thr_y * maxvalue // 0xFF
    thr_u = thr_u * maxvalue // 0xFF
    thr_v = thr_v * maxvalue // 0xFF
    mask_uv = core.std.Expr([core.std.Expr(core.std.ShufflePlanes(mask, 1, GRAY), [f'x {thr_u} < 0 {maxvalue} ?']), core.std.Expr(core.std.ShufflePlanes(mask, 2, GRAY), [f'x {thr_v} < 0 {maxvalue} ?'])], ['x y +']).resize.Point(mask.width, mask.height)
    mask_yuv_on_y = core.std.Expr([core.std.Expr(core.std.ShufflePlanes(mask, 0, GRAY), [f'x {thr_y} < 0 {maxvalue} ?']), mask_uv], ['x y +']).std.Maximum().std.Deflate()
    return mask_yuv_on_y

def ApplyF3kdbGrain(clip: VideoNode, mask: VideoNode = None, sigma: int = 25, tbsize: int = 3, thsad: int = 100, thsadc: int = None, detect_y: int = 80, detect_c: int = None, grain_y: int = 120, grain_c: int = None, dyn_grain: bool = False, tv_range: bool = True) -> VideoNode:
    ''' Some hard deband implementation by me '''
    funcName = 'ApplyF3kdbGrain'
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{funcName}: "clip" must be a clip!')
    if thsadc is None:
        thsadc = thsad
    if detect_c is None:
        detect_c = detect_y
    if grain_c is None:
        grain_c = grain_y
    clip16 = core.fmtc.bitdepth(clip, bits=16)
    if mask is None:
        repairmask = JensenLineMask(clip, 8, 12, 12).std.Deflate().fmtc.bitdepth(bits=16)
    else:
        repairmask = mask.fmtc.bitdepth(bits=16)
    pf = core.dfttest.DFTTest(clip16, sigma=sigma, tbsize=tbsize, planes=[0])
    filtered = cooldegrain.CoolDegrain(clip, tr=1, thsad=thsad, thsadc=thsadc, bits=16, blksize=8, overlap=4, pf=pf)
    filtered = core.f3kdb.Deband(filtered, y=detect_y, cb=detect_c, cr=detect_c, grainy=grain_y, grainc=grain_c, dynamic_grain=dyn_grain, keep_tv_range=tv_range, output_depth=16)
    filtered = core.std.MaskedMerge(filtered, clip16, repairmask, planes=[0,1,2], first_plane=True)
    return filtered

def ProcessRegion(clip: VideoNode, filtering: callable, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0, mask: bool = False) -> VideoNode:
    region = core.std.Crop(clip, left, right, top, bottom)
    filtered = filtering(region)
    padded = core.std.AddBorders(filtered, left, right, top, bottom)
    if mask:
        binaryMask = core.std.Binarize(padded, threshold=1)
        return core.std.MaskedMerge(clip, padded, binaryMask)
    return padded

def MergeHalf(source: VideoNode, filtered: VideoNode, right: bool = True) -> VideoNode:
    ''' Applies filter only to left or right half of frame '''
    funcName = 'MergeHalf'
    if not isinstance(source, VideoNode):
        raise TypeError(f'{funcName}: "source" must be a clip!')
    if not isinstance(filtered, VideoNode):
        raise TypeError(f'{funcName}: "filtered" must be a clip!')
    assert source.num_frames == filtered.num_frames, TypeError(f'{funcName}: input clips are not even! source: {source.num_frames}, filtered: {filtered.num_frames}!')
    if right is True:
        source_part = core.std.CropRel(source, right=source.width//2)
        filtered_part = core.std.CropRel(filtered, left=source.width//2)
        merged = core.std.StackHorizontal([source_part, filtered_part])
    else:
        source_part = core.std.CropRel(source, left=source.width//2)
        filtered_part = core.std.CropRel(filtered, right=source.width//2)
        merged = core.std.StackHorizontal([filtered_part, source_part])
    return merged

def MakeTestEncode(clip: VideoNode) -> VideoNode:
    ''' Selects a few ranges from entire video to examine compression '''
    funcName = 'MakeTestEncode'
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{funcName}: "clip" must be a clip!')
    cycle = int(clip.num_frames / 100 * 2)
    selev = core.std.SelectEvery(clip, cycle=cycle, offsets=range(50))
    selev = core.std.AssumeFPS(selev, fpsnum=clip.fps_num, fpsden=clip.fps_den)
    return selev

def DiffCreditlessMask(titles: VideoNode, nc: VideoNode) -> VideoNode:
    ''' Makes mask based on difference from 2 clips. Raises a mask from that diiference '''
    funcName = 'DiffCreditlessMask'
    if not isinstance(titles, VideoNode):
        raise TypeError(f'{funcName}: "titles" must be a clip!')
    if not isinstance(nc, VideoNode):
        raise TypeError(f'{funcName}: "nc" must be a clip!')
    assert titles.num_frames == nc.num_frames, TypeError(f'{funcName}: input clips are not even! titles: {titles.num_frames}, nc: {nc.num_frames}!')
    test = core.std.MakeDiff(titles, nc, [0])
    test = core.std.ShufflePlanes(test, 0, GRAY)
    test = test.std.Prewitt().std.Expr('x 25 < 0 x ?').std.Expr('x 2 *')
    test = core.rgvs.RemoveGrain(test, 4).std.Expr('x 30 > 255 x ?')
    return test

def DiffRescaleMask(clip: VideoNode, descale_h: int = 720, descale_w: int = None, kernel: str = 'bicubic', b=1/3, c=1/3, taps: int = 3, mode: str = "approx", mthr: int = 55, upscale_thrs: bool = True) -> VideoNode:
    ''' Builds mask from difference of original and re-upscales clips '''
    funcName = 'DiffRescaleMask'
    def str2kernel(kernel: str = 'bicubic'):
        kernels = {
            'bicubic': core.resize.Bicubic,
            'bilinear': core.resize.Bilinear,
            'spline16': core.resize.Spline16,
            'spline36': core.resize.Spline36,
            'lanczos': core.resize.Lanczos
        }
        return kernels[kernel]
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{funcName}: "clip" must be a clip!')
    descale_w = m4((clip.width * descale_h) / clip.height) if descale_w == None else descale_w
    bits = clip.format.bits_per_sample
    maxvalue = (1 << bits) - 1
    half_pixel = 128 * maxvalue // 0xFF
    dclip = dsc.Descale(clip, descale_w, descale_h, kernel=kernel, b=b, c=c)
    upscaler = str2kernel(kernel)
    uclip = upscaler(dclip, clip.width, clip.height, filter_param_a=b if kernel != "lanczos" else taps, filter_param_b=c)
    uclip = core.std.ShufflePlanes(uclip, 0, GRAY)
    clip = core.std.ShufflePlanes(clip, 0, GRAY)
    diff = core.std.MakeDiff(clip, uclip)
    if mode == "approx":
        mask = diff.rgvs.RemoveGrain(2).rgvs.RemoveGrain(2).hist.Luma()
        mask = mask.std.Expr(f'x {mthr} < 0 x ?')
        mask = mask.std.Prewitt().std.Maximum().std.Maximum().std.Deflate()
    elif mode == "precise":
        if len(mthr) == 3:
            mult_1pass, thr, mult_2pass = mthr
        elif len(mthr) == 2:
            mult_1pass, thr = mthr
            mult_2pass = 4
        elif len(mthr) == 1 and mthr != 55: # not default value specified
            mult_1pass, thr, mult_2pass = 10, mthr, 4
        else:
            mult_1pass, thr, mult_2pass = 10, 32, 4
        if upscale_thrs:
            mult_1pass, thr, mult_2pass = mult_1pass * maxvalue // 0xFF, thr * maxvalue // 0xFF, mult_2pass * maxvalue // 0xFF
        mask = core.std.Expr(diff, f'x {half_pixel} - {mult_1pass} *').std.Maximum().std.Expr(f'x {thr} < 0 x {mult_2pass} * ?').std.Inflate()
    else:
        raise ValueError(f'{funcName}: invalid mode.')
    return mask

def DiffOn2FramesMask(clip: VideoNode, first: int = 0, second: int = 0, thr: int = 30, luma_only: bool = True) -> VideoNode:
    ''' Helper for building masks using 2 frames of clip '''
    funcName = 'DiffOn2FramesMask'
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{funcName}: "clip" must be a clip!')
    planes = [0,1,2]
    bits = clip.format.bits_per_sample
    maxvalue = (1 << bits) - 1
    thr = thr * maxvalue // 0xFF
    if luma_only:
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
        planes = [0]
    frame1 = clip[first]
    frame2 = clip[second]
    fmdiff = core.std.MakeDiff(frame1, frame2, planes).std.Sobel(planes=planes).std.Expr(f'x {thr} < 0 x ?')
    return fmdiff

def ApplyMaskOnLuma(source: VideoNode, filtered: VideoNode, mask: VideoNode) -> VideoNode:
    ''' Performs MaskedMerge on bright (luma) component of two clips. Colors will be copied from first clip '''
    funcName = 'ApplyMaskOnLuma'
    if not isinstance(source, VideoNode):
        raise TypeError(f'{funcName}: "source" must be a clip!')
    if not isinstance(filtered, VideoNode):
        raise TypeError(f'{funcName}: "filtered" must be a clip!')
    source_y = core.std.ShufflePlanes(source, 0, GRAY)
    filtered_y = core.std.ShufflePlanes(filtered, 0, GRAY)
    mask_y = core.std.ShufflePlanes(mask, 0, GRAY)
    masked = core.std.MaskedMerge(filtered_y, source_y, mask_y)
    result = core.std.ShufflePlanes([masked, source, source], planes=[0, 1, 2], colorfamily=source.format.color_family)
    return result

def eedi3Scale(input: VideoNode, uheight: int = 720, arx: int = None, ary: int = None, eedi3Mode: str = 'cpu', nnedi3Mode: str = 'cpu', lumaDevice: int = -1, chromaDevice: int = -1, pscrn: int = 1, alpha: float = 0.2, beta: float = 0.25, gamma: float = 1000.0) -> VideoNode:
    ''' Some eedi3-based upscale function. Luma will be upscaled with eedi3+nnedi3 filters, chroma with nnedi3 '''
    funcName = 'eedi3Scale'
    if not isinstance(input, VideoNode):
        raise TypeError(f'{funcName}: "input" must be a clip!')
    def nnedi3_superclip(clip, nnedi3Mode='cpu', device=-1, pscrn=1, dw=False):
        if dw and nnedi3Mode != 'opencl':
            step1 = core.nnedi3.nnedi3(clip, field=1, dh=True, nsize=0, nns=4, pscrn=pscrn)
            rot1  = core.std.Transpose(step1)
            step2 = core.nnedi3.nnedi3(rot1, field=1, dh=True, nsize=0, nns=4, pscrn=pscrn)
            return core.std.Transpose(step2)
        if nnedi3Mode == 'opencl':
            return core.nnedi3cl.NNEDI3CL(clip, field=1, dh=True, dw=dw, nsize=0, nns=4, pscrn=pscrn, device=device)
        elif nnedi3Mode == 'znedi3':
            return core.znedi3.nnedi3(clip, field=1, dh=True, nsize=0, nns=4, pscrn=pscrn)
        else:
            return core.nnedi3.nnedi3(clip, field=1, dh=True, nsize=0, nns=4, pscrn=pscrn)
    def eedi3_instance(clip, eedi3Mode='cpu', nnedi3Mode='cpu', device=-1, pscrn=1, alpha=0.2, beta=0.25, gamma=1000.0):
        if eedi3Mode == 'opencl':
            return core.eedi3m.EEDI3CL(clip, field=1, dh=True, alpha=alpha, beta=beta, gamma=gamma, vcheck=3, sclip=nnedi3_superclip(clip, nnedi3Mode, device, pscrn), device=device)
        else:
            return core.eedi3m.EEDI3(clip, field=1, dh=True, alpha=alpha, beta=beta, gamma=gamma, vcheck=3, sclip=nnedi3_superclip(clip, nnedi3Mode, device, pscrn))
    w = input.width
    h = input.height
    ux = w * 2
    uy = h * 2
    dy = uheight
    if arx and ary:
        dx = m4(dy / ary * arx)
    else:
        dx = m4(dy / h * w)
    cw = dx >> input.format.subsampling_w
    cy = dy >> input.format.subsampling_h
    Y = core.std.ShufflePlanes(input, 0, GRAY)
    U = core.std.ShufflePlanes(input, 1, GRAY)
    V = core.std.ShufflePlanes(input, 2, GRAY)
    Y = eedi3_instance(Y, eedi3Mode, nnedi3Mode, lumaDevice, pscrn, alpha, beta, gamma)
    Y = core.std.Transpose(Y)
    Y = eedi3_instance(Y, eedi3Mode, nnedi3Mode, lumaDevice, pscrn, alpha, beta, gamma)
    Y = core.std.Transpose(Y)
    Y = core.resize.Spline36(Y, dx, dy, src_left=-0.5, src_top=-0.5, src_width=ux, src_height=uy)
    U = core.resize.Spline36(nnedi3_superclip(U, device=chromaDevice, pscrn=pscrn, dw=True), cw, cy, src_left=-0.25, src_top=-0.5)
    V = core.resize.Spline36(nnedi3_superclip(V, device=chromaDevice, pscrn=pscrn, dw=True), cw, cy, src_left=-0.25, src_top=-0.5)
    return core.std.ShufflePlanes([Y, U, V], [0, 0, 0], YUV)

def RfsMany(clip: VideoNode, source: VideoNode, mappings: list = None, my_func: callable = None) -> VideoNode:
    ''' Yet another wrapper for feeding many manual static masks at once. Uses modified rf.Replace function '''
    funcName = 'RfsMany'
    intervals = []
    clips = []
    my_func_args = {}
    assert clip.num_frames == source.num_frames, TypeError(f'{funcName}: input clips are not even! clip: {clip.num_frames}, source: {source.num_frames}!')
    if mappings == None:
        raise ValueError('Not enough parameters.')
    if my_func != None and not callable(my_func):
        raise ValueError('Passed function is not callable.')
    if not isinstance(mappings, list):
        raise ValueError('Mappings holds non-list data.')
    for mapping in mappings:
        if not isinstance(mapping, list):
            raise ValueError('One of mappings iterations holds non-list data.')
        if len(mapping) == 4:
            start, end, arg_vals, my_cust_func = [value for value in mapping]
            if not callable(my_cust_func):
                raise ValueError('Passed custom function is not callable.')
            just_replace = False
        elif len(mapping) == 3:
            start, end, arg_vals = [value for value in mapping]
            my_cust_func = False
            just_replace = False
        elif len(mapping) == 2:
            start, end = [value for value in mapping]
            my_cust_func = False
            just_replace = True
        else:
            raise ValueError('One of mappings lacks some values.')
        if my_func is None and my_cust_func is False and just_replace is False:
            raise ValueError('You should provide at least [start, end] positions for just replacement.')
        try:
            if my_cust_func:
                arg_names = inspect.getargspec(my_cust_func)[0]
            elif just_replace:
                arg_names = []
            else:
                arg_names = inspect.getargspec(my_func)[0]
        except:
            raise ValueError('Something went wrong with passed function.')
        if not just_replace:
            if not isinstance(arg_vals, list):
                arg_vals = [arg_vals]
            arg_vals = [clip, source] + arg_vals
            arg_pos = 0
            for arg_name in arg_names:
                if len(arg_vals) > arg_pos:
                    my_func_args[arg_name] = arg_vals[arg_pos]
                arg_pos += 1
            if my_cust_func:
                clips.append(my_cust_func(**my_func_args))
            else:
                clips.append(my_func(**my_func_args))
        else:
            clips.append(source)
        intervals.append(f'[{start}:{end}]')
    return core.rfmod.Replace(clip, clips, intervals)

def rfs(clipa: VideoNode, clipb: VideoNode, mappings: Sequence = None) -> VideoNode:
    ''' Basically a wrapper for std.Trim and std.Splice that recreates the functionality of
        AviSynth's ReplaceFramesSimple (http://avisynth.nl/index.php/RemapFrames)
        that was part of the plugin RemapFrames by James D. Lin 
        Almost a copypaste from fvsfunc (some minor changes).'''
    if not isinstance(clipa, VideoNode):
        raise TypeError('RFS: "clipa" must be a clip!')
    if not isinstance(clipb, VideoNode):
        raise TypeError('RFS: "clipb" must be a clip!')
    if clipa.format.id != clipb.format.id:
        raise TypeError('RFS: "clipa" and "clipb" must have the same format!')
    if mappings is not None and not isinstance(mappings, Sequence):
        raise TypeError('RFS: "mappings" must be a list!')
    if mappings is None:
        mappings = []

    maps = []
    for item in mappings:
        if isinstance(item, int):
            maps.append([item, item])
        elif isinstance(item, Sequence):
            maps.append(item)

    for start, end in maps:
        if start > end:
            raise ValueError('RFS: Start frame is bigger than end frame: [{} {}]'.format(start, end))
        if end >= clipa.num_frames or end >= clipb.num_frames:
            raise ValueError('RFS: End frame too big, one of the clips has less frames: {}'.format(end)) 

    out = clipa
    for start, end in maps:
        temp = clipb[start:end+1] 
        if start != 0:
            temp = out[:start] + temp
        if end < out.num_frames - 1:
            temp = temp + out[end+1:]
        out = temp
    return out

def retinex_edgemask(src: VideoNode, sigma: int = 1, draft: bool = False, opencl: bool = False, device: int = -1) -> VideoNode:
    '''
    Use retinex to greatly improve the accuracy of the edge detection in dark scenes.
    draft=True is a lot faster, albeit less accurate
    sigma is the sigma of tcanny
    '''
    src = core.fmtc.bitdepth(src, bits=16)
    luma =  core.std.ShufflePlanes(src, 0, GRAY)
    if draft:
        ret = core.std.Expr(luma, 'x 65535 / sqrt 65535 *')
    else:
        ret = core.retinex.MSRCP(luma, sigma=[50, 200, 350], upper_thr=0.005)
    tcanny_clip = core.tcanny.TCannyCL(ret, mode=1, sigma=sigma, device=device) if opencl else core.tcanny.TCanny(ret, mode=1, sigma=sigma)
    mask = core.std.Expr([kirsch(luma), tcanny_clip.std.Minimum(coordinates=[1, 0, 1, 0, 0, 1, 0, 1])], 'x y +')
    return mask

def kirsch(src: VideoNode) -> VideoNode:
    '''
    Kirsch edge detection. This uses 8 directions, so it's slower but better than Sobel (4 directions).
    '''
    w = [5] * 3 + [-3] * 5
    weights = [w[-i:] + w[:-i] for i in range(4)]
    c = [core.std.Convolution(src, (w[:4] + [0] + w[4:]), saturate=False) for w in weights]
    return core.std.Expr(c, 'x y max z max a max')

def m4(x):
    return 16 if x < 16 else int(x // 4 + 0.5) * 4
