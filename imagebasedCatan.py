# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import numpy as np
import scipy.misc as mc
import scipy as sp
import scipy.ndimage
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from copy import deepcopy as dc
from sklearn.cluster import DBSCAN
from operator import itemgetter
import random as rdm
import time

def extract_color(IMGo,thresh = (0.4,0.6,0.05)):
    IMG = mpl.colors.rgb_to_hsv(IMGo)
    uIMG,lIMG = (IMG[:,:,0] >= thresh[0]),(IMG[:,:,0] <= thresh[1])
    sIMG = IMG[:,:,1] <= thresh[2]
    img = np.logical_or(np.logical_and(uIMG,lIMG),sIMG)
    return img

def final_pointer(Dict,call):
    continue_recursion = True
    called = []
    while continue_recursion:
        try:
            prv_call = call
            call = Dict[prv_call]
            if call == prv_call:
                raise KeyError
            called.append(prv_call)
        except KeyError:
            continue_recursion = False
    return call,called

def adjust_dict_pointers(Dict,call_set = None):
    if call_set == None:
        call_set = set(Dict.keys())
    while len(call_set) > 0:
        call = call_set.pop()
        final_point,called = final_pointer(Dict,call)
        for call in called:
            Dict[call] = final_point
            try:
                call_set.remove(call)
            except KeyError:
                pass
    return Dict

def given_coords_connectedComponents(cimg,coord_list,return_additionals=False,
                                     neighbor_size=21):
    Class,equivalence,pair_set,sizes,coords = 2,{},set(),{},{}
    if type(neighbor_size) is not int:
        int(neighbor_size)
    if (not neighbor_size % 2) or (neighbor_size < 3):
        raise AssertionError("neighbor_size must be odd and at least 3")
    half = neighbor_size / 2
    for i,j in coord_list:
        neighbor_matrix = cimg[i-half:i+half+1,j-half:j+half+1]
        u = set(np.unique(neighbor_matrix))
        if 0 in u: u.remove(0)
        if 1 in u: u.remove(1)
        if len(u) == 0:
            cimg[i,j] = Class
            Class +=1
        elif len(u) == 1:
            cimg[i,j] = u.pop()
        else:
            min_class = min(u)
            u.remove(min_class)
            for pair in u:
                try:
                    if min_class < equivalence[pair] :
                        equivalence[pair] = min_class
                except KeyError:
                    equivalence[pair] = min_class
                    pair_set.add(pair)
            cimg[i,j] = min_class
    equivalence = adjust_dict_pointers(equivalence,pair_set)
    for i,j in coord_list:
        Class = cimg[i,j]
        try:
            Class = equivalence[Class]
            cimg[i,j] = Class
        except KeyError:
            pass
        if return_additionals:
            try:
                sizes[Class] += 1
                coords[Class].append((i,j))
            except KeyError:
                sizes[Class] = 1
                coords[Class] = [(i,j)]
    if return_additionals:
        return cimg,sizes,coords
    return cimg

def connectedComponents(img,return_additionals = False):
    m,n = np.shape(img)
    Iimg = np.reshape(np.arange(m*n),(m,n))
    img = img.astype(int)
    linear_coords = np.unique(img * Iimg)
    coords = []
    for coord in linear_coords:
        coords.append(((coord/n),(coord-n*(coord/n))))
    return given_coords_connectedComponents(img,coords,return_additionals,3)

def isolate_component(cimg,sizes,return_id = False,size_rank = 0):
    try:
        sizes.pop(0)
    except KeyError:
        pass
    sizes = sorted(sizes.items(),key=itemgetter(1))[::-1]
    component = sizes[size_rank][0]
    if return_id:
        return cimg == component,component
    return cimg == component

def border_extraction(cimg, coord_list):
    bimg = np.zeros(np.shape(cimg))
    border_coord_list = []
    for i,j in coord_list:
        threebythree = cimg[i-1:i+2,j-1:j+2]
        if (3,3) == np.shape(threebythree):
            if np.any(threebythree == 0):
                bimg[i,j] = 1
                border_coord_list.append((i,j))
        else:
            bimg[i,j] = 1
            border_coord_list.append((i,j))
    return bimg,border_coord_list

def euc(x,y):
    return np.sqrt(sum((x-y)**2.0))

def isolate_closest_component(cimg, coord_list, coord = 'center'):
    if coord == 'center':
        m,n = np.shape(cimg)
        coord = np.array([m/2.0,n/2.0])
    absolute_min_comp, absolute_min_dist = -1,float('inf')
    for component,coords in coord_list.items():
        coords = np.array(coords)
        min_dist = float('inf')
        for c in coords:
            dist = euc(c,coord)
            if dist < min_dist:
                min_dist = dist
        if min_dist < absolute_min_dist:
            absolute_min_dist = min_dist
            absolute_min_comp = component
    return cimg == absolute_min_comp, absolute_min_comp

def linematrix(length, degree, intensity = 255):
    pi = np.pi
    if degree < 0:
        if degree < -2*pi:
            m = np.floor(degree/(-2*pi))
            degree += m*2*pi
        degree += 2*pi
    elif degree > 2*pi:
        m = np.floor(degree/2*pi)
        degree = degree - m*2*pi
    
    if (degree >= (2*pi - pi/4) and degree <= 2*pi) or \
        (degree >= 0 and degree <= pi/4) or \
        (degree >= (pi - pi/4) and degree <= (pi + pi/4)):
        interval = int(round(length * np.abs(np.cos(degree))))
        x = np.arange(interval)
        y = np.round(np.tan(degree)*x)
        
        if min(y) < 0:
            y = abs(min(y)) + y
        
        h = int(max(y))
        lm = np.zeros((h+1,interval))
        for i in range(interval):
            lm[int(h-y[i]),int(x[i])] = intensity
    else:
        interval = int(round(length * np.abs(np.sin(degree))))
        y = np.arange(interval)
        x = np.round(y/np.tan(degree))
        
        if min(x) < 0:
            x = abs(min(x)) + x
        
        w = int(max(x))
        lm = np.zeros((interval,w+1))
        for i in range(interval):
            lm[int(y[i]),int(w-x[i])] = intensity
    return lm

def imfill(test_array,h_max=255):
    input_array = np.copy(test_array) 
    el = sp.ndimage.generate_binary_structure(2,2).astype(np.int)
    inside_mask = sp.ndimage.binary_erosion(~np.isnan(input_array), \
                                            structure=el)
    output_array = np.copy(input_array)
    output_array[inside_mask]=h_max
    output_old_array = np.copy(input_array)
    output_old_array.fill(0)   
    el = sp.ndimage.generate_binary_structure(2,1).astype(np.int)
    while not np.array_equal(output_old_array, output_array):
        output_old_array = np.copy(output_array)
        output_array = np.maximum(input_array,sp.ndimage.grey_erosion(\
                                  output_array, size=(3,3), footprint=el))
    return output_array

def hexagonmatrix(length, degree, fill = True, intensity = 255):
    pi = np.pi
    if degree > pi/3:
        scale = degree / (pi/3)
        rsc = np.floor(scale)
        degree = degree - rsc*(pi/3)
    firstline = linematrix(length,degree,intensity)
    r1,c1 = np.shape(firstline)
    onetwentyline = linematrix(length,degree + (2.0/3.0)*pi,intensity)
    r2,c2 = np.shape(onetwentyline)
    twofortyline = linematrix(length,degree + (4.0/3.0)*pi,intensity)
    r3,c3 = np.shape(twofortyline)
    if degree >= 0 and degree <= pi/6:
        midupappend = np.zeros((r3-1,c1-2))
        leftupappend = np.zeros(((r3-r2)+r1-1,c2))
        rightbotappend = np.zeros((r1-1,c3))
        mid = np.concatenate((midupappend , firstline[:,1:(c1-1)]),0)
        left = np.concatenate((np.concatenate((leftupappend,onetwentyline),0),\
                               mid),1)
        bothalf = np.concatenate((left,
                np.concatenate((twofortyline,rightbotappend),0)),1)
        tophalf = mc.imrotate(bothalf,180)
        rh,ch = np.shape(bothalf)
        shiftpoint = 0
        for i in range(rh):
            if bothalf[i,0] == intensity:
                shiftpoint = i
                break
        hexagon = np.zeros((rh + (rh - shiftpoint),ch))
        hexagon[(rh-shiftpoint):(2*rh-shiftpoint),:] = \
            hexagon[(rh-shiftpoint):(2*rh-shiftpoint),:] + bothalf
        hexagon[0:rh,:] = hexagon[0:rh,:] + tophalf
        hexagon = (hexagon >= intensity)*intensity
        if fill:
            hexagon = imfill(hexagon)
    else:
        hexagon = hexagonmatrix(length,(pi/3 - degree),fill)
        hexagon = np.flip(hexagon,0)
    hexagon = hexagon.astype(float)
    hexagon = (hexagon/np.max(hexagon))*intensity
    return hexagon

def update_accuracy(submatrix,l,lg,accuracy):
    totpoints = float(lg + np.sum(submatrix))
    ia = np.sum((submatrix - l)**2.0)/totpoints
    accuracy.append(ia + 2.0/totpoints)
    return accuracy

def check_line_angle(leng,angle,bimg,m,n,pr,pc,accuracy):
    pi = np.pi
    l = linematrix(leng,angle,1)
    rl,cl = np.shape(l)
    lg = float(np.sum(l))
    if ((pr-rl+1)<0) or ((pc-cl+1)<0) or ((pr+rl)>m) or ((pc+cl)>n):
        accuracy.append(float('inf'))
    elif (angle >= 0) and (angle <= pi/2):
        submatrix = bimg[(pr-rl+1):(pr+1),pc:(pc+cl)]
        accuracy = update_accuracy(submatrix,l,lg,accuracy)
    elif (angle >= pi) and (angle <= 3.0*pi/2.0):
        submatrix = bimg[pr:(pr+rl),(pc-cl+1):(pc+1)]
        accuracy = update_accuracy(submatrix,l,lg,accuracy)
    elif (angle > pi/2.0) and (angle < pi):
        submatrix = bimg[(pr-rl+1):(pr+1),(pc-cl+1):(pc+1)]
        accuracy = update_accuracy(submatrix,l,lg,accuracy)
    elif (angle > 3.0*pi/2.0):
        submatrix = bimg[pr:(pr+rl),pc:(pc+cl)]
        accuracy = update_accuracy(submatrix,l,lg,accuracy)
    return accuracy

def errorlines(bimg,pr,pc,leng,method=180,points=90):
    if (method != 180) and (method != 360):
        raise AssertionError('method must either be 180 or 360')
    pi = np.pi
    m,n = np.shape(bimg)
    if method == 360:
        points *= 2
        deg = np.linspace(0,2*pi,points)
    else:
        deg = np.linspace(0,pi,points)
    accuracy,mindeg,minerror = [],None,float('inf')
    for i in deg:
        accuracy = check_line_angle(leng,i,bimg,m,n,pr,pc,accuracy)
        if i > pi:
            i -= pi
        if accuracy[-1] < minerror:
            mindeg = i
            minerror = accuracy[-1]
        if type == 180:
            j = i + pi
            accuracy = check_line_angle(leng,j,bimg,m,n,pr,pc,accuracy)
            if accuracy[-1] < minerror:
                mindeg = j
                minerror = accuracy[-1]
    return mindeg,minerror

def point_angle(p2y,p1y,p2x,p1x):
    try:
        angle = np.arctan((p2y-p1y)/float(p2x-p1x))
    except ZeroDivisionError:
        angle = np.pi/2.0
    if angle < 0: angle += np.pi
    return angle

def eucMM(M,m):
    if type(M) is list:
        M = np.array(M)
    length,x1,x2 = -float('inf'),None,None
    for i in range(len(M)):
        for j in range(i+1,len(M)):
            d = euc(M[i],M[j])
            if d > length:
                length,x1,x2 = d,M[i],M[j]
    angle = point_angle((m-1)-x2[0],(m-1)-x1[0],x2[1],x1[1])
    center = [int((x1[0]+x2[0])/2.0),int((x1[1]+x2[1])/2.0)]
    return length,angle,center,[x1,x2]

def exists_in_intervals(val,intervals):
    for interval in intervals:
        if (val >= interval[0]) and (val <= interval[1]):
            return True
    return False

def wrap_angle_range(angle):
    return angle - np.floor(angle/(2*np.pi))*2*np.pi

def average_angle(angles):
    angles = np.array(angles)*2.0
    avg_angle = angles[0]
    for i in range(1,len(angles)):
        A = [avg_angle,angles[i]]
        argmx = np.argmax(A)
        DR = A[argmx] - A[argmx-1]
        MR = 2*np.pi - DR
        if DR < MR:
            if argmx:
                avg_angle = A[argmx-1] + (DR/(2+(i-1)))
            else:
                avg_angle = A[argmx] - (DR/(2+(i-1)))
        else:
            if argmx:
                avg_angle = A[argmx-1] - (MR/(2+(i-1)))
            else:
                avg_angle = A[argmx] + (MR/(2+(i-1)))
    return wrap_angle_range(avg_angle)/2.0

def neighbor_cluster(bimg,GP,PI,N=41):
    lsimg,lssizes,lscoords = given_coords_connectedComponents(bimg,GP,True,N)
    m,n = np.shape(bimg)
    L,A,C,E,fL,fA,fC,fE = {},{},{},{},{},{},{},{}
    for ls,coords in lscoords.items():
        try:
            l,a,c,ends = eucMM(coords,m)
        except TypeError:
            continue
        L[ls],A[ls],C[ls],E[ls] = l,a,c,ends
    Lv,Lk,appended = L.values(),L.keys(),set()
    lower_len,upper_len = np.percentile(Lv,50),np.max(Lv)
    for i in range(len(Lk)):
        ls1 = Lk[i]
        ul,ua,uc,ue = L[ls1],A[ls1],C[ls1],E[ls1]
        if (ls1 not in appended):
            for j in range(i+1,len(Lk)):
                ls2 = Lk[j]
                if (L[ls2] < lower_len) and (ls2 not in appended):
                    end_coords = ue+E[ls2]
                    l1,a1,c1,e1 = eucMM(end_coords,m)
                    if exists_in_intervals(a1,PI) and (l1 <= upper_len):
                        ul,ua,uc,ue = l1,a1,c1,e1
                        appended.add(ls2)
        if ul >= lower_len:
            fL[ls1],fA[ls1],fC[ls1],fE[ls1] = ul,ua,uc,ue
    bl,ba = np.percentile(fL.values(),70),average_angle(fA.values())
    return fL,fA,fC,fE,bl,ba

def loop_around(List,index):
    l = len(List)
    if index >= l: index -= l
    elif index < 0: index += l
    return List[index],index

def degree_finding(bimg,bcoord_list,leng=20,pass_rate = 0.25,pts=90,N=61,
                   plot = False):
    prvec,pcvec,mindeg = [],[],[]
    for pr,pc in bcoord_list:
        md,me = errorlines(bimg,pr,pc,leng,360,pts)
        if me < pass_rate:
            prvec.append(pr),pcvec.append(pc),mindeg.append(md)
    counts,intervals = np.histogram(mindeg)
    if plot:
        print("Angles Histogram:")
        plt.hist(mindeg)
        plt.show()
    mx_arg = np.argmax(counts)
    startdeg,finishdeg = intervals[mx_arg],intervals[mx_arg+1]
    potent_intervals = [(startdeg,finishdeg)]
    c3,i3 = loop_around(counts,mx_arg+1)
    c1,i1 = loop_around(counts,mx_arg-1)
    Q3_cutoff = np.percentile(counts,75)
    if c3 >= Q3_cutoff:
        potent_intervals.append((intervals[i3],intervals[i3+1]))
    if c1 >= Q3_cutoff:
        potent_intervals.append((intervals[i1],intervals[i1+1]))
    GP = []
    for i in range(len(prvec)):
        if (mindeg[i] >= startdeg) and (mindeg[i] <= finishdeg):
            GP.append((prvec[i],pcvec[i]))
        else:
            bimg[prvec[i],pcvec[i]] = 0
    bimg = bimg.astype(int)
    L,A,C,E,bl,ba = neighbor_cluster(bimg,GP,potent_intervals,N)
    if plot:
        m,n = np.shape(bimg)
        GP = np.array(GP)
        fig = plt.figure(figsize=(4,4))
        print("Line Segments Found:")
        plt.scatter(GP[:,1],(m-1)-GP[:,0])
        for c in C:
            plt.scatter(C[c][1],(m-1)-C[c][0],marker='s',color='orange')
        plt.show()
    return L,A,C,E,bl,ba

def sign(x):
    if x >= 0: return 1
    return -1

def get_resized_res_imgs(sample_hexagon):
    hr,hc = np.shape(sample_hexagon)
    fns = ['images\\brick.jpg','images\\desert.jpg','images\\gold.jpg','images\\water.jpg','images\\sheep.jpg',
           'images\\wheat.jpg','images\\wood.jpg','images\\ore.jpg']
    D = {}
    for fn in fns:
        name = fn[:-4]
        res = scipy.misc.imresize(mplimg.imread(fn),(hr,hc))
        for i in range(3):
            res[:,:,i] *= sample_hexagon
        D[name] = res
    return D

def get_resized_num_imgs(sample_hexagon,ratio=1/3.0):
    hr,hc = np.shape(sample_hexagon)
    nr,nc = int(hr*ratio),int(hc*ratio)
    nums = [2,3,4,5,6,8,9,10,11,12]
    fns = ['images\\'+str(num)+'.jpg' for num in nums]
    D = {}
    for i in range(len(fns)):
        numb = scipy.misc.imresize(mplimg.imread(fns[i]),(nr,nc))
        D[nums[i]] = numb
    return D

def place_number(img,numb,NB,hexagon):
    if numb == None:
        return img
    if type(hexagon) is tuple:
        pr,pc = hexagon
    else:
        pr,pc = hexagon.pos_r,hexagon.pos_c
    nr,nc,o = np.shape(NB[numb])
    img[(pr+nr):(pr+2*nr),(pc+nc):(pc+2*nc),:] = NB[numb]
    if type(hexagon) is not tuple:
        hexagon.number = numb
    return img

def hexagon_distance(h1,h2):
    return np.sqrt((h1.pos_r-h2.pos_r)**2+(h1.pos_c-h2.pos_c)**2)

def reset_hex_neighbors(hexagons):
    for i in range(len(hexagons)):
        hexagons[i].neighboring_hexes = []

def get_hex_neighbors(hexagons):
    for i in range(len(hexagons)):
        h1 = hexagons[i]
        h1.index = i
        for j in range(i+1,len(hexagons)):
            h2 = hexagons[j]
            if hexagon_distance(h1,h2) < 1.4*np.sqrt(3)*h1.length:
                h1.neighboring_hexes.append(j)
                h2.neighboring_hexes.append(i)
    
def replace_with_res(img,unit_hexagon,hexagon,RD,hidden=False):
    pr,pc,hr,hc = hexagon.pos_r,hexagon.pos_c,hexagon.hr,hexagon.hc
    resname = hexagon.res_type if not hidden else 'water'
    res_img = RD[resname]
    for i in range(3):
        submatrix = img[pr:(pr+hr),pc:(pc+hc),i]
        submatrix -= unit_hexagon*submatrix
    img[pr:(pr+hr),pc:(pc+hc),:] += res_img
    if not hidden:
        hexagon.res_type = resname
    return img

def numb2score(numb):
    if numb <= 6:
        return numb - 1
    return numb*-1 + 13

def pdf2cdf(pdf):
    cdf,c = np.array([]),0.0
    for p in pdf:
        cdf = np.append(cdf,c+p)
        c += p
    return cdf/cdf[-1]

def pick_rdm_cdf_idx(cdf):
    r = rdm.random()
    for i in range(len(cdf)):
        if r <= cdf[i]:
            break
    return i

def rdm_int_tri_dist(min_int,max_int):
    rdm_seq = []
    rev_seq = range(min_int,max_int+1)[::-1]
    i = 1
    for integer in rev_seq:
        for times in range(i):
            rdm_seq.append(integer)
        i += 1
    return rdm.choice(rdm_seq)

def radial_recurser(hexagon,hexagons,R,iL):
    untouched_neighbors = []
    for neighbor_id in hexagon.neighboring_hexes:
        if neighbor_id in R:
            untouched_neighbors.append(neighbor_id)
            iL.append(neighbor_id)
            R.remove(neighbor_id)
    for r in untouched_neighbors:
        R,iL = radial_recurser(hexagons[r],hexagons,R,iL)
    return R,iL

def rdm_radial_distr(hexagons,as_indexes=False):
    r = np.random.choice(len(hexagons))
    R = set(range(len(hexagons)))
    R.remove(r)
    R,iL = radial_recurser(hexagons[r],hexagons,R,[r])
    if as_indexes:
        return iL
    new_hex_arrange = []
    for i in iL:
        new_hex_arrange.append(hexagons[i])
    return new_hex_arrange

def rdm_arrange(L,as_indexes=False):
    rdm_int = rdm.choice(range(len(L)))
    irng = range(rdm_int,len(L))+range(rdm_int)
    if as_indexes:
        return irng
    new_L = []
    for i in irng:
        new_L.append(L[i])
    return new_L

def uni_rdm_arrange(L,as_indexes=False):
    R = np.random.choice(len(L),len(L),False)
    if as_indexes:
        return R
    new_L = []
    for i in R:
        new_L.append(L[i])
    return new_L

def distribute_resources(Ctn):
    R = Resources(Ctn.num_of_hexes)
    for hexagon in rdm_arrange(Ctn.hexagons):
        R.pick_resource(hexagon,Ctn.hexagons)
    Ctn.R = R

def paint_resources(Ctn):
    nIMG = dc(Ctn.IMG)
    for hexagon in Ctn.hexagons:
        nIMG = replace_with_res(nIMG,Ctn.unit_hex,hexagon,Ctn.Res)
    return nIMG

def distribute_numbers(Ctn):
    for hexagon in rdm_radial_distr(Ctn.hexagons):
        if (hexagon.res_type != 'water') and (hexagon.res_type != 'desert'):
            Ctn.R.pick_number(hexagon,Ctn.hexagons)
        
def paint_numbers(IMG,Ctn):
    for hexagon in Ctn.hexagons:
        IMG = place_number(IMG,hexagon.number,Ctn.Nums,hexagon)
    return IMG

def plot_board(IMG,mn_ratio = 8.0):
    m,n,o = np.shape(IMG)
    if m > n:
        fz = (mn_ratio,mn_ratio*(m/float(n)))
    else:
        fz = (mn_ratio*(n/float(m)),mn_ratio)
    fig = plt.figure(figsize=fz)
    ax = fig.add_subplot(111)
    ax.imshow(IMG)
    plt.show()
    
def clear_Catan(Ctn):
    Ctn.IMG = mplimg.imread(Ctn.img_file)
    for hexagon in Ctn.hexagons:
        hexagon.res_type = None
        hexagon.number = None
        
def revealed_board(Ctn):
    retry = 'yes'
    while retry == 'yes':
        clear_Catan(Ctn)
        distribute_resources(Ctn)
        nIMG = paint_resources(Ctn)
        plot_board(nIMG)
        retry=input('Re-distribute? Type yes if so, otherwise anything: ')
    retry = 'yes'
    while retry == 'yes':
        distribute_numbers(Ctn)
        nIMG = paint_numbers(nIMG,Ctn)
        plot_board(nIMG)
        retry=input('Re-distribute? Type yes if so, otherwise anything: ')

def get_digit_image(number,nr,nc):
    img = mplimg.imread('images\\'+number+'.jpg')
    img = scipy.misc.imresize(img,(nr,nc))
    img[:,:,2] = 0.0
    return img
        
def get_unplotted_num(number, sample_hexagon,ratio = 1.0/3.0):
    hr,hc = np.shape(sample_hexagon)
    sn = str(number)
    nr,nc = int(hr*ratio),int((hc*ratio)/len(sn))
    init_image = get_digit_image(sn[0],nr,nc)
    if len(sn) == 1:
        return init_image
    for i in range(1,len(sn)):
        new_image = get_digit_image(sn[i],nr,nc)
        init_image = np.concatenate((init_image,new_image),1)
    return init_image

def place_unplotted_number(img,numb,hexagon):
    pr,pc = hexagon.pos_r,hexagon.pos_c
    numb_img = get_unplotted_num(numb,hexagon.hexmat)
    nr,nc,o = np.shape(numb_img)
    img[(pr+nr):(pr+2*nr),(pc+nc):(pc+2*nc),:] = numb_img
    return img

def hidden_board(Ctn):
    hidden_indexes = set(range(len(Ctn.hexagons)))
    prmpt = 'Choose yellow tile number to reveal (or type quit to exit): '
    clear_Catan(Ctn)
    distribute_resources(Ctn)
    distribute_numbers(Ctn)
    nIMG = dc(Ctn.IMG)
    for i in range(len(Ctn.hexagons)):
        hexagon = Ctn.hexagons[i]
        nIMG = replace_with_res(nIMG,Ctn.unit_hex,hexagon,Ctn.Res,'water')
        nIMG = place_unplotted_number(nIMG,i,hexagon)
    plot_board(nIMG)
    print("You have the following options: ")
    print("1) Type quit to exit out of the program.")
    print("2) Type an integer of a yellow number hex to reveal it.")
    print("3) To reveal multiple hexes at a time, separate the integers "+\
    "with a comma with no space inbetween (e.g. 3,4,6)")
    print("4) Type all to reveal the remaining hidden hexes.")
    while len(hidden_indexes) != 0:
        reveal_index = input(prmpt)
        if reveal_index == 'quit':
            break
        elif reveal_index == 'all':
            for i in hidden_indexes:
                hexagon = Ctn.hexagons[i]
                nIMG = replace_with_res(nIMG,Ctn.unit_hex,hexagon,Ctn.Res)
                nIMG = place_number(nIMG,hexagon.number,Ctn.Nums,hexagon)
            plot_board(nIMG)
            break
        indexes = reveal_index.split(',')
        for hex_index in indexes:
            try:
                i = int(hex_index)
            except ValueError:
                print("Unrecognized value entered. Make sure it is an "+\
                "integer which has not been revealed yet. Try again... ")
                continue
            if i not in hidden_indexes:
                print("Either this number has already been revealed or it "+\
                "is not on the board. Try again... ")
                continue
            hexagon = Ctn.hexagons[i]
            nIMG = replace_with_res(nIMG,Ctn.unit_hex,hexagon,Ctn.Res)
            nIMG = place_number(nIMG,hexagon.number,Ctn.Nums,hexagon)
            hidden_indexes.remove(i)
        plot_board(nIMG)

class Resources:
    def __init__(self,total_res,style = 'home'):
        self.res_amt_o = {'desert':5,'gold':4}
        lands = set(['brick','wood','wheat','ore','sheep'])
        if total_res <= 15:
            num_of_land = 3
        elif total_res <= 20:
            num_of_land = 4
        elif total_res <= 25:
            num_of_land = 5
        else:
            num_of_land = 6
        for land in lands: self.res_amt_o[land] = num_of_land
        self.numb_amt = {2:1,3:2,4:2,5:2,6:2,8:2,9:2,10:2,11:2,12:1}
        self.add_amt = {2:1,3:1,4:1,5:1,6:1,8:1,9:1,10:1,11:1,12:1}
        self.res_amt,tots = {},0
        for res in self.res_amt_o:
            if res == 'desert':
                n = 0
            elif res == 'gold':
                n = rdm_int_tri_dist(1,self.res_amt_o[res])
            else:
                n = rdm_int_tri_dist(3,self.res_amt_o[res])
            self.res_amt[res] = n
            tots += n
        self.res_amt['water'] = total_res - tots
        self.res_nms =['ore','brick','sheep','wood','wheat',
                       'gold','desert','water']
        self.res_funcs = {'ore':self.lnd_func,'brick':self.lnd_func,
                'sheep':self.lnd_func,'wood':self.lnd_func,
                'wheat':self.lnd_func,'gold':self.gold_function,
                'desert':self.dsrt_func,'water':self.water_function}
        self.nmbs = [2,3,4,5,6,8,9,10,11,12]
        self.nmb_res = {i:{nm:0 for nm in self.res_nms} for i in self.nmbs}
    def water_function(self,hexagon,land,hexagons):
        w = 0.0
        for Ni in hexagon.neighboring_hexes:
            N = hexagons[Ni]
            if N.res_type == 'water': w+=1
        return (1/8.0)**((5.0/6.0)**w)
    def gold_function(self,hexagon,land,hexagons):
        lands = set(['ore','brick','sheep','wood','wheat','desert'])
        w,l = 0.0,0.0
        for Ni in hexagon.neighboring_hexes:
            N = hexagons[Ni]
            if N.res_type == 'water': w += 1
            elif N.res_type in lands: l += 1
        return (1/8.0)**((5.0/6.0)**w) * (2/3.0)**l
    def lnd_func(self,hexagon,land,hexagons):
        l,w = 0.0,0.0
        for Ni in hexagon.neighboring_hexes:
            N = hexagons[Ni]
            if N.res_type == land: l += 1
            elif N.res_type == 'water': w += 1
        p = 1.5 if w > 2 else 1
        return (1/8.0) * (0.5)**l * p
    def dsrt_func(self,hexagon,land,hexagons):
        return 0.0
    def keep_tracker(self,numb):
        if np.sum(self.numb_amt.values()) <= 0:
            self.numb_amt = dc(self.add_amt)
        keep = 1 if self.numb_amt[numb] > 0 else 0
        return keep
    def same_number(self,hexagon,numb,hexagons):
        s = 0.0
        for Ni in hexagon.neighboring_hexes:
            N = hexagons[Ni]
            if N.number == numb: s += 1
        pa = (0.5)**s*(3.0/5.0)**(self.nmb_res[numb][hexagon.res_type])
        pa *= self.keep_tracker(numb)
        return pa
    def se(self,hexagon,numb,hexagons):
        s,k = 0.0,{n:0 for n in range(1,6)}
        for Ni in hexagon.neighboring_hexes:
            N = hexagons[Ni]
            if N.number == numb: s += 1
            if N.number != None:
                k[numb2score(N.number)] += 1
        self.p5 = (0.4)**(1+0.05*k[1]+0.1*k[2]+0.2*k[3]+0.8*k[4]+2.0*k[5])
        pa = self.p5*(0.5)**s*(3.0/5.0)**(self.nmb_res[numb][hexagon.res_type])
        pa *= self.keep_tracker(numb)
        return pa
    def tt(self,hexagon,numb,hexagons):
        s = 0.0
        for Ni in hexagon.neighboring_hexes:
            N = hexagons[Ni]
            if N.number == numb: s += 1
        self.p1 = (4.0**(-10.0*self.p5+1))/10.0
        pa = self.p1*(0.5)**s*(3.0/5.0)**(self.nmb_res[numb][hexagon.res_type])
        pa *= self.keep_tracker(numb)
        return pa
    def fn(self,hexagon,numb,hexagons):
        s = 0.0
        for Ni in hexagon.neighboring_hexes:
            N = hexagons[Ni]
            if N.number == numb: s += 1
        if self.p5 <= 0.1:
            self.p4 = self.p5
        elif self.p5 <= 0.2:
            self.p4 = -15*(self.p5-0.2)**2 + 0.25
        else:
            self.p4 = -np.sqrt((self.p5-0.2)/(80.0/9.0)) + 0.25
        pa = self.p4*(0.5)**s*(3.0/5.0)**(self.nmb_res[numb][hexagon.res_type])
        pa *= self.keep_tracker(numb)
        return pa
    def te(self,hexagon,numb,hexagons):
        s = 0.0
        for Ni in hexagon.neighboring_hexes:
            N = hexagons[Ni]
            if N.number == numb: s += 1
        if self.p1 <= 0.1:
            self.p2 = self.p1
        elif self.p1 <= 0.2:
            self.p2 = -15*(self.p1-0.2)**2 + 0.25
        else:
            self.p2 = -np.sqrt((self.p1-0.2)/(80.0/9.0)) + 0.25
        pa = self.p2*(0.5)**s*(3.0/5.0)**(self.nmb_res[numb][hexagon.res_type])
        pa *= self.keep_tracker(numb)
        return pa
    def ft(self,hexagon,numb,hexagons):
        s = 0.0
        for Ni in hexagon.neighboring_hexes:
            N = hexagons[Ni]
            if N.number == numb: s += 1
        self.p3 = np.mean([self.p2,self.p4])
        pa = self.p3*(0.5)**s*(3.0/5.0)**(self.nmb_res[numb][hexagon.res_type])
        pa *= self.keep_tracker(numb)
        return pa
    def pick_resource(self,hexagon,hexagons):
        pdf = []
        for nm in self.res_nms:
            if self.res_amt[nm] <= 0:
                pdf.append(0.0)
            else:
                pdf.append(self.res_funcs[nm](hexagon,nm,hexagons))
        cdf = pdf2cdf(pdf)
        i = pick_rdm_cdf_idx(cdf)
        hexagon.res_type = self.res_nms[i]
        self.res_amt[self.res_nms[i]] -= 1
    def pick_number(self,hexagon,hexagons):
        p6 = self.se(hexagon,6,hexagons)
        p8 = self.p5*self.same_number(hexagon,8,hexagons)
        p2 = self.tt(hexagon,2,hexagons)
        p12 = self.p1*self.same_number(hexagon,12,hexagons)
        p5 = self.fn(hexagon,5,hexagons)
        p9 = self.p4*self.same_number(hexagon,9,hexagons)
        p3 = self.te(hexagon,3,hexagons)
        p11 = self.p2*self.same_number(hexagon,11,hexagons)
        p4 = self.ft(hexagon,4,hexagons)
        p10 = self.p3*self.same_number(hexagon,10,hexagons)
        pdf = [p2,p3,p4,p5,p6,p8,p9,p10,p11,p12]
        cdf = pdf2cdf(pdf)
        i = pick_rdm_cdf_idx(cdf)
        hexagon.number = self.nmbs[i]
        self.numb_amt[self.nmbs[i]] -= 1
        self.nmb_res[self.nmbs[i]][hexagon.res_type] += 1

class Hexagon:
    def __init__(self,pos_r,pos_c,length,angle):
        self.pos_r,self.pos_c,self.length,self.angle = \
        pos_r,pos_c,length,angle
        self.res_type = None
        self.number = None
        self.neighboring_hexes = []
    def gen_hexmat(self,intensity=0.5,fill=True):
        self.hexmat = hexagonmatrix(self.length,self.angle,fill,intensity)
        self.hr,self.hc = np.shape(self.hexmat)
    def get_hexmat(self,with_dimensions=False,intensity=0.5,fill=True):
        try:
            if with_dimensions:
                return self.hexmat,self.hr,self.hc
            return self.hexmat
        except AttributeError:
            self.gen_hexmat(intensity,fill)
            if with_dimensions:
                return self.hexmat,self.hr,self.hc
            return self.hexmat

class Catan:
    def __init__(self,img_file='images\\standard.jpg',plt_steps=True,write_steps=True,
                 thresh = (0.4,0.6,0.05),border_neighborhood=71,
                 segment_neighborhood=61,i_leng=20,pass_rate=0.25,points=120,
                 i_stepsize=0.5,dispersion=1.01,hex_id=0.5,
                 tile_threshold=0.15,immediate_build=True):
        self.img_file,self.write_steps = img_file,write_steps
        self.plot_steps = plt_steps
        self.thresh,self.i_leng,self.pass_rate = thresh,i_leng,pass_rate
        self.border_neighborhood,self.points = border_neighborhood,points
        self.segment_neighborhood,self.hex_id = segment_neighborhood,hex_id
        self.i_stepsize,self.dispersion = i_stepsize,dispersion
        self.tile_threshold = tile_threshold
        self.p_type = {'left':0,'up left':3,'down left':2,'down right':1,
             'right':4,'up right':5}
        self.hexagons,self.unplotted_hexes,self.num_of_hexes = [],set(),0
        self.potential_hexes = set()
        if immediate_build: self.build_Catan()
    def build_Catan(self):
        if self.write_steps:
            s = time.time()
            print("Extracting image...")
        self.IMG = mplimg.imread(self.img_file)
        img = extract_color(self.IMG,self.thresh)
        self.m,self.n = np.shape(img)
        if self.plot_steps:
            print("HSV < Threshold:")
            plt.imshow(img)
            plt.show()
        if self.write_steps:
            print("Running Connected Components... (< 3 minutes inshaa Allah)")
        cimg,sizes,coords = connectedComponents(img,True)
        self.cimg,component = isolate_component(cimg, sizes, True)
        coord_list = coords[component]
        if self.plot_steps:
            print("Largest Isolated Component:")
            plt.imshow(self.cimg)
            plt.show()
        if self.write_steps:
            print("Extracting inner borders...")
        bimg,bcoord_list = border_extraction(self.cimg, coord_list)
        if self.plot_steps:
            print("Extracted Borders: ")
            bcl0 = np.array(bcoord_list)
            fig = plt.figure(figsize=(4,4))
            plt.scatter(bcl0[:,1],(self.m-1)-bcl0[:,0])
            plt.show()
        bimg,new_sizes,new_coords = given_coords_connectedComponents(bimg,
                                    bcoord_list,True,self.border_neighborhood)
        bimg,bcomp = isolate_component(bimg,new_sizes,True,1)
        bcoord_list = new_coords[bcomp]
        if self.plot_steps:
            print("Inner borders (2nd largest component):")
            bcl2 = np.array(bcoord_list)
            fig = plt.figure(figsize=(4,4))
            plt.scatter(bcl2[:,1],(self.m-1)-bcl2[:,0])
            plt.show()
        if self.write_steps:
            print("Computing hexagon lengths and angles... "+\
                  "(< 5 minutes inshaa Allah)")
        self.L,self.A,self.C,self.E,self.bl,self.ba = \
            degree_finding(bimg,bcoord_list,self.i_leng,self.pass_rate,
                         self.points,self.segment_neighborhood,self.plot_steps)
        self.himg = self.cimg.astype(float)/4.0
        sample_hex = hexagonmatrix(self.bl,self.ba,True,1.0)
        self.tile_error_thresh = self.tile_threshold*np.sum(sample_hex)
        self.hr,self.hc = np.shape(sample_hex)
        self.hm = sample_hex * self.hex_id
        self.find_border_hexes()
        if self.plot_steps:
            print("Border Hexagons:")
            plt.imshow(self.himg)
        self.find_hex_hexes()
        if self.plot_steps:
            print("Full Hexagon Fill:")
            plt.imshow(self.himg)
        if self.write_steps:
            print("Runtime: "+str((time.time()-s)/60.0)+" minutes")
        self.unit_hex = sample_hex.astype('uint8')
        self.Res = get_resized_res_imgs(self.unit_hex)
        self.Nums = get_resized_num_imgs(self.unit_hex)
        get_hex_neighbors(self.hexagons)
    def tile_placing_old(self,position,pr,pc,dispersion=1.01):
        shdg, shift = self.ba - np.pi/2.0, self.bl*np.sqrt(3)*dispersion
        c = sign(shdg)
        deg_adjust = [0,2*np.pi/3,-2*np.pi/3,-np.pi/3,0,np.pi/3]
        col_const,row_const = [-1,-1,-1,1,1,1],[c,1,-1,-1,-c,1]
        p = self.p_type[position]
        adjrow,adjcol = np.shape(linematrix(int(dispersion*shift),
                                            shdg+deg_adjust[p]))
        nextj,nexti = pc + col_const[p]*(adjcol-1),pr + row_const[p]*(adjrow-1)
        submat = dc(self.himg[nexti:(nexti+self.hr),nextj:(nextj+self.hc)])
        if (((nexti+self.hr) <= self.m) and (nexti >= 0)) and \
            (((nextj+self.hc) <= self.n) and (nextj >= 0)):
                submat += self.hm
        else:
            return float('inf'),nexti,nextj
        error = np.sum(submat > self.hex_id) #- np.sum(submat == (1.0/4.0))
        return error,nexti,nextj
    def tile_placing(self,pr,pc,use_thresh_method=False):
        dispersion = self.dispersion if use_thresh_method else self.i_stepsize
        shdg, shift = self.ba - np.pi/2.0, self.bl*np.sqrt(3)*dispersion
        hexagons,min_error,m_r,m_c = [],float('inf'),None,None
        for deg in np.linspace(shdg,shdg+(5*np.pi/3),6):
            nexti = -(shift*np.sin(deg))+pr
            nextj = shift*np.cos(deg)+pc
            nexti = int(np.floor(nexti)) if nexti < pr else int(np.ceil(nexti))
            nextj = int(np.floor(nextj)) if nextj < pc else int(np.ceil(nextj))
            submat = dc(self.himg[nexti:(nexti+self.hr),nextj:(nextj+self.hc)])
            if (((nexti+self.hr) <= self.m) and (nexti >= 0)) and \
            (((nextj+self.hc) <= self.n) and (nextj >= 0)):
                submat += self.hm
                error = np.sum(submat > self.hex_id)
            else:
                error = float('inf')
            if use_thresh_method and (error < self.tile_error_thresh):
                hexagons.append(Hexagon(nexti,nextj,self.bl,self.ba))
            if error < min_error:
                min_error,m_r,m_c = error,nexti,nextj
        if not use_thresh_method:
            hexagons = Hexagon(m_r,m_c,self.bl,self.ba)
        return hexagons
    def update_hex_list(self,new_hexagon,potential = True):
        self.hexagons.append(new_hexagon)
        if potential:
            self.potential_hexes.add(self.num_of_hexes)
        self.unplotted_hexes.add(self.num_of_hexes)
        self.num_of_hexes += 1
    def find_border_hexes(self):
        for pr,pc in self.C.values():
            c_r,c_c = int(pr-self.hr/2.0),int(pc-self.hc/2.0)
            self.update_hex_list(self.tile_placing(c_r,c_c))
            self.place_empty_hexagons()
        disperts = []
        for i in range(len(self.hexagons)):
            h1 = self.hexagons[i]
            for j in range(i+1,len(self.hexagons)):
                h2 = self.hexagons[j]
                disperts.append(hexagon_distance(h1,h2)/(np.sqrt(3)*self.bl))
        disperts = np.array(disperts)
        disperts = disperts[disperts < 1.4]
        if len(disperts) > 0:
            self.dispersion = np.mean(disperts)
            if self.dispersion < 1:
                self.dispersion = np.min(disperts)
    def find_hex_hexes(self):
        while len(self.potential_hexes) != 0:
            new_hex = []
            while len(self.potential_hexes) != 0:
                hex_id = self.potential_hexes.pop()
                hexagon = self.hexagons[hex_id]
                pr,pc = hexagon.pos_r,hexagon.pos_c
                new_hex += self.tile_placing(pr,pc,True)
            if self.write_steps:
                print("Number of potential hexes before collision: "+\
                str(len(new_hex)))
            collided_already = set()
            for i in range(len(new_hex)):
                if i in collided_already:
                    continue
                h1 = new_hex[i]
                cllsns = 0.0
                for j in range(i+1,len(new_hex)):
                    h2 = new_hex[j]
                    dist = hexagon_distance(h1,h2)
                    if dist < 0.5*np.sqrt(3)*self.bl:
                        cllsns += 1.0
                        collided_already.add(j)
                        h1.pos_r = (h1.pos_r*cllsns + h2.pos_r)/(cllsns+1.0)
                        h1.pos_c = (h1.pos_c*cllsns + h2.pos_c)/(cllsns+1.0)
                h1.pos_r = int(h1.pos_r)
                h1.pos_c = int(h1.pos_c)
                self.update_hex_list(h1)
            self.place_empty_hexagons()
    def place_empty_hexagons(self):
        for hex_i in self.unplotted_hexes:
            h = self.hexagons[hex_i]
            hexmat,hr,hc = h.get_hexmat(True,self.hex_id,True)
            self.himg[h.pos_r:(h.pos_r+hr),h.pos_c:(h.pos_c+hc)] += hexmat
        self.unplotted_hexes = set()
    def retry_hex_placement(self,dispersion=None,tile_threshold=None):
        self.himg = self.cimg.astype(float)/4.0
        self.hexagons,self.unplotted_hexes,self.num_of_hexes = [],set(),0
        self.potential_hexes = set()
        sample_hex = hexagonmatrix(self.bl,self.ba,True,1.0)
        if tile_threshold != None:
            self.tile_error_thresh = tile_threshold*np.sum(sample_hex)
        self.find_border_hexes()
        if dispersion != None:
            self.dispersion = dispersion
        self.find_hex_hexes()
        plt.imshow(self.himg)
    def play(self,hidden = True):
        if hidden:
            hidden_board(self)
        else:
            revealed_board(self)