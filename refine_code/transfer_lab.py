import cv2
import numpy as np
import math
import copy
ngrid = 32
def calculate_singlegrid(ori_r, ori_g, ori_b):
    
    ntmp = ngrid + 1
    ntmpsqr = ntmp * ntmp

    tmpx = ori_r / 255 * ngrid
    diff_x = tmpx - math.floor(tmpx)
    tmpx = math.floor(tmpx)
    if (tmpx == ngrid) :
        tmpx = ngrid - 1
        diff_x = 1
    tmpy = ori_g / 255 * ngrid
    diff_y = tmpy - math.floor(tmpy)
    tmpy = math.floor(tmpy)
    if (tmpy == ngrid) :
        tmpy = ngrid - 1
        diff_y = 1
    tmpz = ori_b / 255 * ngrid
    diff_z = tmpz - math.floor(tmpz)
    tmpz = math.floor(tmpz)
    if (tmpz == ngrid) :
        tmpz = ngrid - 1
        diff_z = 1

    corner_ind = tmpz * ntmpsqr + tmpy * ntmp + tmpx

    
    res = [0 for i in range(16)]

    res[0] = corner_ind
    res[1] = corner_ind + ntmpsqr
    res[2] = corner_ind + ntmp
    res[3] = corner_ind + ntmp + ntmpsqr
    res[4] = corner_ind + 1
    res[5] = corner_ind + ntmpsqr + 1
    res[6] = corner_ind + ntmp + 1
    res[7] = corner_ind + ntmp + ntmpsqr + 1

    res[8] = (1 - diff_x) * (1 - diff_y) * (1 - diff_z)
    res[9] = (1 - diff_x) * (1 - diff_y) * diff_z
    res[10] = (1 - diff_x) * diff_y * (1 - diff_z)
    res[11] = (1 - diff_x) * diff_y * diff_z
    res[12] = diff_x * (1 - diff_y) * (1 - diff_z)
    res[13] = diff_x * (1 - diff_y) * diff_z
    res[14] = diff_x * diff_y * (1 - diff_z)
    res[15] = diff_x * diff_y * diff_z
    return res

def prepare_grid(img_rgb, lable):
    global grid_size,ngrid
    grid_size = (ngrid + 1) * (ngrid + 1) * (ngrid + 1)
    global grid_lab ,weightindex,weightmap
    global grid_R,grid_G,grid_B
    grid_lab = [None for i in range(grid_size)]
    grid_R = [None for i in range(grid_size)]
    grid_G = [None for i in range(grid_size)]
    grid_B = [None for i in range(grid_size)]
    step = 255.0 / ngrid
    tot = 0

    for i in range(0,ngrid+1):
        for j in range(0, ngrid+1):
            for k in range(0, ngrid+1):
                grid_lab[tot] = RGB2LAB([k * step, j * step, i * step])
                # grid_lab[tot] = [k * step, j * step, i * step]
                tot += 1
    valid_idx_y, valid_idx_x = np.where(lable != 0)
    weightindex = [None for i in range(len(valid_idx_x))]
    weightmap = [None for i in range(len(valid_idx_x))]
    for idx,(y,x) in enumerate(zip(valid_idx_y,valid_idx_x)):
            res = calculate_singlegrid(img_rgb[y,x,0],img_rgb[y,x,1],img_rgb[y,x,2])
            weightindex[idx] = [0 for t in range(8)]
            weightmap[idx] = [0 for t in range(8)]
            for j in range(8):
                weightindex[idx][j] = res[j]
            for j in range(8):
                weightmap[idx][j] = res[j+8]
    return grid_lab, weightindex, weightmap

def calculate_LAB_distance(l1, a1, b1, l2, a2, b2):
    l1 = float(l1)
    a1 = float(a1)
    b1 = float(b1)
    l2 = float(l2)
    a2 = float(a2)
    b2 = float(b2)
    K1 = 0.045
    K2 = 0.015
    del_L = l1 - l2
    c1 = math.sqrt(a1 * a1 + b1 * b1)
    c2 = math.sqrt(a2 * a2 + b2 * b2)
    c_ab = c1 - c2
    h_ab = (a1 - a2) * (a1 - a2) + (b1 - b2) * (b1 - b2) - c_ab * c_ab
    return del_L * del_L + c_ab * c_ab / (1 + K1 * c1) / (1 + K1 * c1) + h_ab / (1 + K2 * c1) / (1 + K2 * c1)

def big_change(dir):
    return (abs(dir[0]) > 0.5) or (abs(dir[1]) > 0.5) or (abs(dir[2]) > 0.5)

def out_boundary(testrgb):
    out_threshold = 0.2
    return (testrgb[0] < -out_threshold or testrgb[0] > 255 + out_threshold or
            testrgb[1] < -out_threshold or testrgb[1] > 255 + out_threshold or
            testrgb[2] < -out_threshold or testrgb[2] > 255 + out_threshold)

def out_boundary_2(testrgb):
    out_threshold = 0.2
    if testrgb[0] < out_threshold:
        return True, 1
    if testrgb[0] < out_threshold:
        return True, 2
    if testrgb[0] < out_threshold:
        return True, 3
    if testrgb[0] < 255 + out_threshold:
        return True, 4
    if testrgb[0] < 255 + out_threshold:
        return True, 5
    if testrgb[0] < 255 + out_threshold:
        return True, 6
    return False, 0
def rgb_interpolate(bg_color_rgb,ed_color_rgb,steps = 2):
    step_R = (float(ed_color_rgb[0]) - float(bg_color_rgb[0])) / steps
    step_G = (float(ed_color_rgb[1]) - float(bg_color_rgb[1])) / steps
    step_B = (float(ed_color_rgb[2]) - float(bg_color_rgb[2])) / steps
    re = []
    for i in range(steps):
        re.append([int(bg_color_rgb[0]+step_R*i),int(bg_color_rgb[1]+step_G*i),int(bg_color_rgb[2]+step_B*i)])
    return re
def lab2rgb_opencv(c):
    c = np.array(c,dtype=np.uint8)
    c = np.reshape(c,(1,-1,3))
    c = cv2.cvtColor(c,cv2.COLOR_LAB2RGB)
    # print(c)
    return list(c.flatten())

def rgb2lab_opencv(c):
    c = np.array(c,dtype=np.uint8)
    c = np.reshape(c,(1,-1,3))
    c = cv2.cvtColor(c,cv2.COLOR_RGB2LAB)
    return list(c.flatten())
def rgb2hue_opencv(c):
    c = np.array(c,dtype=np.uint8)
    c = np.reshape(c,(1,-1,3))
    c = cv2.cvtColor(c,cv2.COLOR_RGB2HLS)
    return list(c.flatten())

def hue2rgb_opencv(c):
    c = np.array(c,dtype=np.uint8)
    c = np.reshape(c,(1,-1,3))
    c = cv2.cvtColor(c,cv2.COLOR_HLS2RGB)
    return list(c.flatten())
def RGB2LAB(Q):
    _R = ( Q[0] / 255 )
    _G = ( Q[1] / 255 )
    _B = ( Q[2] / 255 )
    
    
    if (_R > 0.04045):
        _R = math.pow(( _R + 0.055 ) / 1.055, 2.4)
    else: 
        _R = _R / 12.92
    
    if (_G > 0.04045):
        _G = math.pow(( _G + 0.055 ) / 1.055, 2.4)
    else:
        _G = _G / 12.92
    
    if (_B > 0.04045):
        _B = math.pow(( _B + 0.055 ) / 1.055, 2.4)
    else:
        _B = _B / 12.92
    
    _R = _R * 100
    _G = _G * 100
    _B = _B * 100
    
    X = _R * 0.4124 + _G * 0.3576 + _B * 0.1805
    Y = _R * 0.2126 + _G * 0.7152 + _B * 0.0722
    Z = _R * 0.0193 + _G * 0.1192 + _B * 0.9505
    
    _X = X / 95.047
    _Y = Y / 100
    _Z = Z / 108.883
    
    if (_X > 0.008856):
        _X = math.pow(_X, 1 / 3)
    else:
        _X = ( 7.787 * _X ) + ( 16 / 116 )
    
    if (_Y > 0.008856):
        _Y = math.pow(_Y, 1 / 3)
    else:
        _Y = ( 7.787 * _Y ) + ( 16 / 116 )
    
    if (_Z > 0.008856):
        _Z = math.pow(_Z, 1 / 3)
    else:
        _Z = ( 7.787 * _Z ) + ( 16 / 116 )
    
    L = ( 116 * _Y ) - 16
    A = 500 * ( _X - _Y )
    B = 200 * ( _Y - _Z )
    
    return [L, A, B]


def LAB2RGB(Q):
    _Y = ( Q[0] + 16 ) / 116
    _X = (Q[1]) / 500 + _Y
    _Z = _Y - (Q[2]) / 200

    if (_Y > 0.206893034422):
        _Y = math.pow(_Y, 3)
    else:
        _Y = ( _Y - 16 / 116 ) / 7.787
    if (_X > 0.206893034422):
        _X = math.pow(_X, 3)
    else:
        _X = ( _X - 16 / 116 ) / 7.787
    if (_Z > 0.206893034422):
        _Z = math.pow(_Z, 3)
    else:
        _Z = ( _Z - 16 / 116 ) / 7.787

    X = 95.047 * _X
    Y = 100 * _Y
    Z = 108.883 * _Z

    _X = X / 100
    _Y = Y / 100
    _Z = Z / 100

    _R = _X * 3.2406 + _Y * -1.5372 + _Z * -0.4986
    _G = _X * -0.9689 + _Y * 1.8758 + _Z * 0.0415
    _B = _X * 0.0557 + _Y * -0.2040 + _Z * 1.0570

    if (_R > 0.0031308):
        _R = 1.055 * math.pow(_R, 1 / 2.4) - 0.055
    else:
        _R = 12.92 * _R
    if (_G > 0.0031308):
        _G = 1.055 * math.pow(_G, 1 / 2.4) - 0.055
    else:
        _G = 12.92 * _G
    if (_B > 0.0031308):
        _B = 1.055 * math.pow(_B, 1 / 2.4) - 0.055
    else:
        _B = 12.92 * _B

    R = _R * 255
    G = _G * 255
    B = _B * 255
    return [R, G, B]


def PURE_LAB2RGB(Q):
    _Y = ( Q[0]*100/255.0 + 16 ) / 116
    _X = (Q[1]-128.0) / 500 + _Y
    _Z = _Y - (Q[2]-128.0) / 200

    if (_Y > 0.206893034422):
        _Y = math.pow(_Y, 3)
    else:
        _Y = ( _Y - 16 / 116 ) / 7.787
    if (_X > 0.206893034422):
        _X = math.pow(_X, 3)
    else:
        _X = ( _X - 16 / 116 ) / 7.787
    if (_Z > 0.206893034422):
        _Z = math.pow(_Z, 3)
    else:
        _Z = ( _Z - 16 / 116 ) / 7.787

    X = 95.0456 * _X
    Y = 100 * _Y
    Z = 108.8754 * _Z

    _X = X / 100
    _Y = Y / 100
    _Z = Z / 100

    _R = _X * 3.2406 + _Y * -1.5372 + _Z * -0.4986
    _G = _X * -0.9689 + _Y * 1.8758 + _Z * 0.0415
    _B = _X * 0.0557 + _Y * -0.2040 + _Z * 1.0570

    if (_R > 0.0031308):
        _R = 1.055 * math.pow(_R, 1 / 2.4) - 0.055
    else:
        _R = 12.92 * _R
    if (_G > 0.0031308):
        _G = 1.055 * math.pow(_G, 1 / 2.4) - 0.055
    else:
        _G = 12.92 * _G
    if (_B > 0.0031308):
        _B = 1.055 * math.pow(_B, 1 / 2.4) - 0.055
    else:
        _B = 12.92 * _B

    R = _R * 255
    G = _G * 255
    B = _B * 255
    return [R, G, B]

def find_boundary(vsrc, dir, l ,r):
    mid = None
    for iter in range(0,30):
        mid = 0.5 * (l + r)
        testrgb= LAB2RGB([vsrc[0] + mid * dir[0], vsrc[1] + mid * dir[1], vsrc[2] + mid * dir[2]] )
        if out_boundary(testrgb):
            r = mid
        else:
            l = mid
    return l



def calculate_single_point(param, matrixsize, oldpalette_L, oldpalette_A, oldpalette_B, diffpalette_L, diffpalette_A, diffpalette_B, vsrc):
    tmpMat = np.zeros((matrixsize, matrixsize),dtype=np.float)
    for u in range(matrixsize):
        for v in range(matrixsize):
            r = calculate_LAB_distance(oldpalette_L[u], oldpalette_A[u], oldpalette_B[u], oldpalette_L[v], oldpalette_A[v], oldpalette_B[v])
            tmpMat[u,v]=math.exp(-r*param)

    tmpD = np.zeros((matrixsize),dtype=np.float)
    for u in range(matrixsize):
        r = calculate_LAB_distance(oldpalette_L[u], oldpalette_A[u], oldpalette_B[u], vsrc[0], vsrc[1], vsrc[2])
        tmpD[u] = math.exp(-r * param)

    precompute_pinv = np.dot(np.linalg.inv(tmpMat), tmpD)
    delta_L = 0
    delta_A = 0
    delta_B = 0

    scale = 0
    for j in range(matrixsize):
        scale = scale + max([precompute_pinv[j],0.0])
    # print('scale',scale)
    for j in range(matrixsize):
        if precompute_pinv[j] > 0:
            delta_L = abs(delta_L) + abs(precompute_pinv[j] / scale * diffpalette_L[j])
            delta_A = delta_A + precompute_pinv[j] / scale * diffpalette_A[j]
            delta_B = delta_B + precompute_pinv[j] / scale * diffpalette_B[j]
    return [vsrc[0] + delta_L, vsrc[1] + delta_A, vsrc[2] + delta_B]
    # return [ delta_L,  delta_A,  delta_B]

def calculate_grid_re(palette_colors, palette_size, oldpalette_L, oldpalette_A, oldpalette_B, diffpalette_L, diffpalette_A, diffpalette_B):
    global grid_R,grid_G,grid_B
    tot = 0
    totr = 0
    RBF_param_coff = 5
    for u in range(palette_size):
        for v in range(u+1,palette_size):
            r = calculate_LAB_distance(oldpalette_L[u], oldpalette_A[u], oldpalette_B[u], oldpalette_L[v],
                                          oldpalette_A[v], oldpalette_B[v])
            tot = tot + 1
            totr = totr + math.sqrt(r)
    if palette_size > 1:
        totr = totr / tot
    else:
        totr = 1.0
    param = RBF_param_coff / (totr*totr)
    palette_change = {}
    for i in range(grid_size):
        vsrc = grid_lab[i]
        tdiff_L = np.zeros((palette_size),dtype=np.float)
        tdiff_A = np.zeros((palette_size),dtype=np.float)
        tdiff_B = np.zeros((palette_size),dtype=np.float)
        for j in range(palette_size):
            dir = [diffpalette_L[j], diffpalette_A[j], diffpalette_B[j]]
            if big_change(dir):
                tttc = [vsrc[0] + dir[0], vsrc[1] + dir[1], vsrc[2] + dir[2]]
                pc = LAB2RGB(tttc)
                flag, ll = out_boundary_2(pc)
                if flag:
                    if ll <= 3 and ll > 0:
                            tttc[ll] -= 0.5 * dir[ll - 1]
                    else:
                        tttc[ll-4] -= 0.5 * dir[ll - 4]
                pc = LAB2RGB(tttc)
                if out_boundary(pc):
                    # print("越界")
                    M = [oldpalette_L[j] + dir[0], oldpalette_A[j] + dir[1], oldpalette_B[j] + dir[2]]
                    Mdir = [vsrc[0] - oldpalette_L[j], vsrc[1] - oldpalette_A[j], vsrc[2] - oldpalette_B[j]]

                    t1 = find_boundary(M, Mdir, 0, 1)
                    t2 = find_boundary([oldpalette_L[j], oldpalette_A[j], oldpalette_B[j]], dir, 1, 300)

                    tdiff_L[j] = abs(dir[0] - (1 - t1) * Mdir[0])
                    tdiff_A[j] = (dir[1] - (1 - t1) * Mdir[1]) / t2
                    tdiff_B[j] = (dir[2] - (1 - t1) * Mdir[2]) / t2
                else:
                    t1 = find_boundary(vsrc, dir, 1, 300)
                    t2 = find_boundary([oldpalette_L[j], oldpalette_A[j], oldpalette_B[j]], dir, 1, 300)

                    lambda1 = min([t1 / t2, 1.0])
                    tdiff_L[j] = abs(diffpalette_L[j])
                    tdiff_A[j] = diffpalette_A[j] * lambda1
                    tdiff_B[j] = diffpalette_B[j] * lambda1
            else:
                tdiff_L[j] = abs(diffpalette_L[j])
                tdiff_A[j] = diffpalette_A[j]
                tdiff_B[j] = diffpalette_B[j]
        res = calculate_single_point(param, palette_size, oldpalette_L, oldpalette_A, oldpalette_B, tdiff_L, tdiff_A,
                                        tdiff_B, vsrc)
        pc = LAB2RGB(res)
        # pc = res
        grid_R[i] = pc[0]
        grid_G[i] = pc[1]
        grid_B[i] = pc[2]

        grid_R[i] = max([0, min([grid_R[i], 255])])
        grid_G[i] = max([0, min([grid_G[i], 255])])
        grid_B[i] = max([0, min([grid_B[i], 255])])

def smoothl(x,d):
    lambda1= 0.2 * math.log(2)
    print("smoothl", math.log( math.exp( lambda1 * x) + math.exp( lambda1 * d) - 1 ) / lambda1 - x)
    return math.log( math.exp( lambda1 * x) + math.exp( lambda1 * d) - 1 ) / lambda1 - x
def modify_newpalette(source_dom_colors, tar_dom_colors):
    def same(a, b):
        return abs(a[0]-b[0]) > 0.5 or  abs(a[1]-b[1]) > 0.5 or  abs(a[2]-b[2]) > 0.5
    ttar = copy.deepcopy(source_dom_colors)
    ttar2 = copy.deepcopy(source_dom_colors)
    print(source_dom_colors,tar_dom_colors)
    for idx,(sc,tc) in enumerate(zip(ttar2,tar_dom_colors)):
        print("idx", idx, sc, tc)

        if same(sc,tc):
            delta_l = float(sc[0]) - float(tc[0])
            print("delta", delta_l)
            for i,c in enumerate(source_dom_colors):
                if i != idx:
                    if source_dom_colors[i][0] < source_dom_colors[idx][0]:
                        print("before ",source_dom_colors[i][0], smoothl(delta_l, source_dom_colors[idx][0]-source_dom_colors[i][0]))
                        source_dom_colors[i][0] = tc[0] - smoothl(delta_l, source_dom_colors[idx][0]-source_dom_colors[i][0])
                    else:
                        source_dom_colors[i][0] = tc[0] + smoothl(-delta_l, source_dom_colors[i][0]-source_dom_colors[idx][0])
                        print("after", source_dom_colors)

            source_dom_colors[idx] = tar_dom_colors[idx]
            print("source_dom_clors 1111",(ttar))
            print("source_dom_clors 2222",(source_dom_colors))
    return ttar, source_dom_colors



def transfer_color(img_lab, img_rgb, lable,palette_colors, source_dom_colors, tar_dom_colors):
    print('source dom colors',(source_dom_colors))
    print('tar_dom_colors',(tar_dom_colors))
    img_rgb_save = copy.deepcopy(img_rgb)
    # source_dom_colors, tar_dom_colors =  modify_newpalette(source_dom_colors,tar_dom_colors)
    prepare_grid(img_rgb, lable)
    oldpalette_L = []
    oldpalette_A = []
    oldpalette_B = []
    diffpalette_L = []
    diffpalette_A = []
    diffpalette_B = []

    for c in source_dom_colors:
        oldpalette_L.append(float(c[0]))
        oldpalette_A.append(float(c[1]))
        oldpalette_B.append(float(c[2]))
    for idx,(sc,tc) in enumerate(zip(source_dom_colors,tar_dom_colors)):
        if idx < len(source_dom_colors) - 1:
            diffpalette_L.append(float(tc[0]) - float(sc[0]))
            diffpalette_A.append(float(tc[1]) - float(sc[1]) - (tar_dom_colors[idx+1][1] - tc[1])*0.5)
            diffpalette_B.append(float(tc[2]) - float(sc[2]) - (tar_dom_colors[idx+1][2] - tc[2])*0.4)
        else:
            diffpalette_L.append(float(tc[0]) - float(sc[0]))
            diffpalette_A.append(float(tc[1]) - float(sc[1]) - (tar_dom_colors[idx ][1] - tar_dom_colors[idx-1 ][1] ) / 2)
            diffpalette_B.append(float(tc[2]) - float(sc[2]) - (tar_dom_colors[idx ][2] - tar_dom_colors[idx-1 ][2]) / 2)
    print(oldpalette_L)
    print('diff', diffpalette_L)
    calculate_grid_re(palette_colors,len(source_dom_colors),oldpalette_L, oldpalette_A, oldpalette_B, diffpalette_L, diffpalette_A, diffpalette_B)
    print(lable.shape)
    valid_idx_y, valid_idx_x = np.where(lable != 0)
    for i, (y, x) in enumerate(zip(valid_idx_y, valid_idx_x)):
        tmpR = 0
        tmpG = 0
        tmpB = 0
        for k in range(8):
            # print('wi',weightindex[i])
            # print('wm',weightmap[i])
            # print(img_rgb[y,x])
            # exit()
            tmpR = tmpR + grid_R[weightindex[i][k]] * weightmap[i][k]
            tmpG = tmpG + grid_G[weightindex[i][k]] * weightmap[i][k]
            tmpB = tmpB + grid_B[weightindex[i][k]] * weightmap[i][k]
        img_rgb[y, x, 0] = tmpR
        img_rgb[y, x, 1] = tmpG
        img_rgb[y, x, 2] = tmpB
    for y, x in zip(valid_idx_y, valid_idx_x):
        mask = max(img_rgb_save[y,x])/255.0
        if mask > 0.7:
            img_rgb[y,x] = mask*img_rgb[y,x] + (1-mask)*img_rgb_save[y,x]
    return img_rgb
