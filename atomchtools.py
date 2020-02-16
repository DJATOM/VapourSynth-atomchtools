from vapoursynth import core, VideoNode, YUV, GRAY
import collections
import cooldegrain
import descale as dsc
import inspect

try:
    from collections.abc import Sequence
except AttributeError:
    from collections import Sequence

__version__ = 0.8

'''
Atomch Tools
Version 0.8 from 16.02.2020

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

def ApplyCredits(credits: VideoNode, nc: VideoNode, fixed_nc: VideoNode, lumaOnly: bool = True) -> VideoNode:
    ''' Convenient helper for applying credits on processed creditless videos (OP/ED) '''
    funcName = 'ApplyCredits'
    if not isinstance(credits, VideoNode):
        raise TypeError(f'{funcName}: "credits" must be a clip!')
    if not isinstance(nc, VideoNode):
        raise TypeError(f'{funcName}: "nc" must be a clip!')
    if not isinstance(fixed_nc, VideoNode):
        raise TypeError(f'{funcName}: "fixed_nc" must be a clip!')
    assert credits.num_frames == nc.num_frames == fixed_nc.num_frames, TypeError(f'{funcName}: input clips are not even! credits: {credits.num_frames}, nc - {nc.num_frames}, fixed_nc - {fixed_nc.num_frames}!')
    if lumaOnly is True:
        credits_ = core.std.ShufflePlanes(credits, 0, GRAY)
        nc = core.std.ShufflePlanes(nc, 0, GRAY)
        fixed_nc = core.std.ShufflePlanes(fixed_nc, 0, GRAY)
    else:
        credits_ = credits
    averaged = core.std.Expr([credits_, nc, fixed_nc], ['x y - z +'])
    if lumaOnly is True:
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

def ApplyImageMask(source: VideoNode, replacement: VideoNode, imageMask: str = None, lumaOnly: bool = True, binarizeThr: int = 128, preview: bool = False) -> VideoNode:
    ''' Applies custom (hand-drawn) image as static mask for two clips '''
    funcName = 'ApplyImageMask'
    if not isinstance(source, VideoNode):
        raise TypeError(f'{funcName}: "source" must be a clip!')
    if not isinstance(replacement, VideoNode):
        raise TypeError(f'{funcName}: "replacement" must be a clip!')
    filemask = core.imwrif.Read(imageMask).resize.Point(format=source.format.id, matrix_s="709", chromaloc_s="top_left")
    NumPlanes = source.format.num_planes
    if lumaOnly is True or NumPlanes == 1:
        planes = [0]
        filemask = core.std.ShufflePlanes(filemask, 0, GRAY)
        source_ = core.std.ShufflePlanes(source, 0, GRAY)
        replacement_ = core.std.ShufflePlanes(replacement, 0, GRAY)
    else:
        planes = [0,1,2]
        source_ = source
        replacement_ = replacement
    assert source.num_frames == replacement.num_frames, TypeError(f'{funcName}: input clips are not even! source: {source.num_frames}, replacement: {replacement.num_frames}!')
    mask = core.std.Binarize(filemask, binarizeThr).std.Maximum().std.Deflate()
    if preview:
        replacement_ = core.std.Merge(mask, replacement_, 0.5)
    masked = core.std.MaskedMerge(source_, replacement_, mask, planes)
    if lumaOnly is True and NumPlanes > 1:
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

def JensenLineMask(clip: VideoNode, thr_y: int = 7, thr_u: int = 8, thr_v: int = 8, scale: int = 1, rg: bool = True) -> VideoNode:
    ''' A modified one to Jensen's needs '''
    funcName = 'JensenLineMask'
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{funcName}: "clip" must be a clip!')
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
    filtered = core.f3kdb.Deband(filtered, dither_algo=3, y=detect_y, cb=detect_c, cr=detect_c, grainy=grain_y, grainc=grain_c, dynamic_grain=dyn_grain, keep_tv_range=tv_range, output_depth=16)
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

def DiffRescaleMask(clip: VideoNode, descale_h: int = 720, descale_w: int = None, kernel: str = 'bicubic', b=1/3, c=1/3, mthr: int = 55) -> VideoNode:
    ''' Builds mask from difference of original and re-upscales clips '''
    funcName = 'DiffRescaleMask'
    def str2kernel(kernel: str = 'bicubic'):
        kernels = {
            'bicubic': core.resize.Bicubic,
            'bilinear': core.resize.Bilinear,
            'spline16': core.resize.Spline16,
            'spline36': core.resize.Spline36
        }
        return kernels[kernel]
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{funcName}: "clip" must be a clip!')
    descale_w = m4((clip.width * descale_h) / clip.height) if descale_w == None else descale_w
    dclip = dsc.Descale(clip, descale_w, descale_h, kernel=kernel, b=b, c=c)
    upscaler = str2kernel(kernel)
    uclip = upscaler(dclip, clip.width, clip.height, filter_param_a=b, filter_param_b=c)
    uclip = core.std.ShufflePlanes(uclip, 0, GRAY)
    clip = core.std.ShufflePlanes(clip, 0, GRAY)
    diff = core.std.MakeDiff(clip, uclip)
    mask = diff.rgvs.RemoveGrain(2).rgvs.RemoveGrain(2).hist.Luma()
    mask = mask.std.Expr(f'x {mthr} < 0 x ?')
    mask = mask.std.Prewitt().std.Maximum().std.Maximum().std.Deflate()
    return mask

def DiffOn2FramesMask(clip: VideoNode, first: int = 0, second: int = 0, thr: int = 30, LumaOnly: bool = True) -> VideoNode:
    ''' Helper for building masks using 2 frames of clip '''
    funcName = 'DiffOn2FramesMask'
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{funcName}: "clip" must be a clip!')
    planes = [0,1,2]
    bits = clip.format.bits_per_sample
    maxvalue = (1 << bits) - 1
    thr = thr * maxvalue // 0xFF
    if LumaOnly:
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
        if dw:
            nnedi3Mode = 'opencl'
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

def RfsMany(clip: VideoNode, source: VideoNode, mappings: list = None, myFunc: callable = None) -> VideoNode:
    ''' Yet another wrapper for feeding many manual static masks at once. Uses modified rf.Replace function '''
    funcName = 'RfsMany'
    intervals = []
    clips = []
    myFuncArgs = {}
    assert clip.num_frames == source.num_frames, TypeError(f'{funcName}: input clips are not even! clip: {clip.num_frames}, source: {source.num_frames}!')
    if mappings == None:
        raise ValueError('Not enough parameters.')
    if myFunc != None and not callable(myFunc):
        raise ValueError('Passed function is not callable.')
    if not isinstance(mappings, list):
        raise ValueError('Mappings holds non-list data.')
    for mapping in mappings:
        if not isinstance(mapping, list):
            raise ValueError('One of mappings iterations holds non-list data.')
        if len(mapping) == 4:
            start, end, argVals, myCustFunc = [value for value in mapping]
            if not callable(myCustFunc):
                raise ValueError('Passed custom function is not callable.')
            justReplace = False
        elif len(mapping) == 3:
            start, end, argVals = [value for value in mapping]
            myCustFunc = False
            justReplace = False
        elif len(mapping) == 2:
            start, end = [value for value in mapping]
            myCustFunc = False
            justReplace = True
        else:
            raise ValueError('One of mappings lacks some values.')
        if myFunc is None and myCustFunc is False and justReplace is False:
            raise ValueError('You should provide at least [start, end] positions for just replacement.')
        try:
            if myCustFunc:
                argNames = inspect.getargspec(myCustFunc)[0]
            elif justReplace:
                argNames = []
            else:
                argNames = inspect.getargspec(myFunc)[0]
        except:
            raise ValueError('Something went wrong with passed function.')
        if not justReplace:
            if not isinstance(argVals, list):
                argVals = [argVals]
            argVals = [clip, source] + argVals
            argPos = 0
            for argName in argNames:
                if len(argVals) > argPos:
                    myFuncArgs[argName] = argVals[argPos]
                argPos += 1
            if myCustFunc:
                clips.append(myCustFunc(**myFuncArgs))
            else:
                clips.append(myFunc(**myFuncArgs))
        else:
            clips.append(source)
        intervals.append(f'[{start}:{end}]')
    return core.rfmod.Replace(clip, clips, intervals)

def rfs(clipa: VideoNode, clipb: VideoNode, mappings: list = None) -> VideoNode:
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
    if mappings is not None and not isinstance(mappings, list):
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

def retinex_edgemask(src: VideoNode, sigma=1, draft=False, openCL: bool = False, device: int = -1) -> VideoNode:
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
    tcannyClip = core.tcanny.TCannyCL(ret, mode=1, sigma=sigma, device=device) if openCL else core.tcanny.TCanny(ret, mode=1, sigma=sigma)
    mask = core.std.Expr([kirsch(luma), tcannyClip.std.Minimum(coordinates=[1, 0, 1, 0, 0, 1, 0, 1])], 'x y +')
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