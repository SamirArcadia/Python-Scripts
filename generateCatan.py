# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:20:44 2020

@author: samir
"""

import numpy as np
import pandas as pd
from copy import deepcopy as dc
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle
from PIL import Image
#from shapely.geometry import box, Point
#from shapely.geometry.polygon import Polygon as Poly
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
#from descartes import PolygonPatch
import gc
import time

"""
There are three gameModes accepted:
    'complete' - Completely fills in the hexes after the border is defined using the resource and number restrictions provided.
    'systematic' - Fill in hexes one step at a time, allowing the user to redraw as they see fit.
    'hidden' - Hide hexes after completely filled in. The user then clicks on the hexes they with to reveal.
"""

hexColors = {None          : {'fc':'none','ec':'gray'},
             'edge'        : {'fc':'none','ec':'gray'},
             'soft border' : {'fc':'C0','ec':'k'},
             'border'      : {'fc':'darkblue','ec':'darkblue','alpha':1,'num':0},
             'outside'     : {'fc':'none','ec':'none'},
             'water'       : {'fc':'skyblue','ec':'none','alpha':1,'num':0},
             'desert'      : {'fc':'burlywood','ec':'none','alpha':1,'num':0},
             'gold'        : {'fc':'goldenrod','ec':'none','alpha':1,'num':None},
             'brick'       : {'fc':'firebrick','ec':'none','alpha':1,'num':None},
             'ore'         : {'fc':'slategray','ec':'none','alpha':1,'num':None},
             'wood'        : {'fc':'forestgreen','ec':'none','alpha':1,'num':None},
             'wheat'       : {'fc':'yellow','ec':'none','alpha':1,'num':None},
             'sheep'       : {'fc':'lawngreen','ec':'none','alpha':1,'num':None}}

resRatios = {'standard' : {'water':0, 'desert':1, 'gold':0, 'brick':3, 'ore':3, 'wood':4, 'wheat':4, 'sheep':4},
             'pangea'   : {'water':0, 'desert':0, 'gold':1, 'brick':3, 'ore':3, 'wood':4, 'wheat':4, 'sheep':4},
             'seafarers' : {'water':20,'desert':1, 'gold':2, 'brick':4, 'ore':4, 'wood':4, 'wheat':4, 'sheep':5}}

resAffinity = pd.DataFrame([[ 8,  1, 30, -4,  3,  0,  0,  0,  0,  0],
                            [-2, -4, -4, -8,  0, -1, -1, -1, -1, -1],
                            [-2,  1,  0, -1,-12, -8, -8, -8, -8, -8],
                            [ 1,  0,  0,  0, -2,-99,  0,  0,  0,  0],
                            [ 1,  0,  0,  0, -2,  0,-99,  0,  0,  0],
                            [ 1,  0,  0,  0, -2,  0,  0,-99,  0,  0],
                            [ 1,  0,  0,  0, -2,  0,  0,  0,-99,  0],
                            [ 1,  0,  0,  0, -2,  0,  0,  0,  0,-99]],
                index=['water','desert','gold','brick','ore','wood','wheat','sheep'],
                columns=[None, 'border','water','desert','gold','brick','ore','wood','wheat','sheep'])

landTiles = ['brick', 'ore', 'wood', 'wheat', 'sheep', 'gold', 'desert']
baseProbs = [ 3/17  ,  3/17,  3/17 ,  3/17  ,  3/17  ,  0 ,  0   ]
landTilesSet = {landTiles[i]:i for i in range(len(landTiles))}
land2waterRatio = 0.5
numTokens = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
numProbs = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
numTokensSet = {numTokens[i]:i for i in range(len(numTokens))}
numWeight = {2:1,3:2,4:3,5:4,6:5,8:5,9:4,10:3,11:2,12:1}

resFiles = {'brick':'images\\brick.jpg','desert':'images\\desert.JPG','gold':'images\\gold.JPG','ore':'images\\ore.jpg',
            'sheep':'images\\sheep.jpg','water':'images\\water.JPG','wheat':'images\\wheat.jpg','wood':'images\\wood.JPG'}

numAffinity = pd.DataFrame([[ 0,  0,  9,-99,  1,  4,  9, 16, 16,  9,  4,  1,  0],
                            [ 1,  1, 16,  1,-99,  1,  4,  9,  9,  4,  1,  0,  1],
                            [ 4,  4,  9,  4,  1,-99,  1,  4,  4,  1,  0,  1,  4],
                            [ 9,  1,  0,  9,  4,  1,-99,  1,  1,  0,  1,  4,  9],
                            [16,  0,-99, 16,  9,  4,  1,-99,  0,  1,  4,  9, 16],
                            [16,  0,-99, 16,  9,  4,  1,  0,-99,  1,  4,  9, 16],
                            [ 9,  1,  0,  9,  4,  1,  0,  1,  1,-99,  1,  4,  9],
                            [ 4,  4,  9,  4,  1,  0,  1,  4,  4,  1,-99,  1,  4],
                            [ 1,  1, 16,  1,  0,  1,  4,  9,  9,  4,  1,-99,  1],
                            [ 0,  0,  9,  0,  1,  4,  9, 16, 16,  9,  4,  1,-99]],
                index=[2, 3, 4, 5, 6, 8, 9, 10, 11, 12],
                columns=[0, None, 'gold', 2, 3, 4, 5, 6, 8, 9, 10, 11, 12])

harborRatio = ['Any\n3:1','Any\n3:1','Any\n3:1','Any\n3:1','Wood\n2:1','Brick\n2:1','Sheep\n2:1','Wheat\n2:1','Ore\n2:1']

calcTotalHarbors = lambda landTiles: max([len(harborRatio), int(landTiles / 2.5)])

def harborPatch(xy, thetaStart=0., width=0.5, height=0.5, resolution=50, **kwargs):
    theta = np.linspace(thetaStart, thetaStart+np.pi, resolution)
    points = np.vstack((width*np.cos(theta)  + xy[0], height*np.sin(theta) + xy[1]))
    return Polygon(points.T, closed=True, **kwargs)

def sample2total(L, total, as_list=True):
    wholeL = np.tile(L, total // len(L))
    remainderL = np.random.choice(L, total % len(L), False)
    S = list(np.concatenate((np.random.choice(wholeL, len(wholeL), False), remainderL)))
    if as_list == True: return S
    unique = L if type(as_list) is bool else as_list
    D = {u:0 for u in unique}
    for u in S: D[u] += 1
    return D

def rand_from_cdf(cdf):
    randomNumber, chosenIndex = np.random.randint(1, np.max(cdf)+1), 0
    while randomNumber > cdf.iloc[chosenIndex]: chosenIndex += 1
    return cdf.index[chosenIndex]

def rand_from_pdf(pdf):
    cdf, choice = [0], 0
    for i in range(len(pdf)): cdf.append(cdf[-1]+pdf[i])
    p = np.random.rand()*cdf[-1] # pick random number between 0 and max of cdf
    while p > cdf[choice+1]: choice += 1
    return choice

class Button:
    def __init__(self, xy, gameboard, name, func, width=0.8, height=0.6, radio=False):
        self.gameboard = gameboard
        fcs = {'Reset':'lavender',  'Reveal':'lightyellow', 'Continent':'lightgoldenrodyellow', 'Redraw':'lightcyan',     'Next':'lightgreen', 'Back':'antiquewhite', 'Radio':'lavender'}
        ecs = {'Reset':'slategrey', 'Reveal':'olive',       'Continent':'olive'               , 'Redraw':'lightseagreen', 'Next':'darkgreen',  'Back':'olive',        'Radio':'orange'}
        tcs = {'Reset':'orangered', 'Reveal':'y',           'Continent':'olivedrab'           , 'Redraw':'darkcyan',      'Next':'green',      'Back':'orange',       'Radio':'orange'}
        self.button_type = 'Radio' if radio else name
        if self.button_type not in fcs: self.button_type = 'Reset' # If an invalid name is entered then default to Reset.
        self.button_func = func
        self.active = False
        offsetx, offsety = width, height
        self.button = Rectangle(xy, offsetx, offsety, fc=fcs[self.button_type], ec=ecs[self.button_type], lw=0, alpha=0.8, zorder=2)
        self.gameboard.ax.add_patch(self.button)
        self.text = self.gameboard.ax.text(xy[0]+offsetx/2, xy[1]+offsety/2, name, fontsize=10, color=tcs[self.button_type], va='center', ha='center', zorder=3)
        self.cidpress = self.button.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidmove = self.button.figure.canvas.mpl_connect('motion_notify_event', self.on_move)
    def on_press(self, event):
        if event.inaxes != self.button.axes: return
        contains, attrd = self.button.contains(event)
        if not contains: return
        self.button_func()
    def on_move(self, event):
        if event.inaxes != self.button.axes: return
        contains, attrd = self.button.contains(event)
        if contains and (not self.active):
            self.button.set_lw(2)
            self.active = True
            self.button.figure.canvas.draw()
        elif (not contains) and self.active: 
            self.button.set_lw(0)
            self.active = False
            self.button.figure.canvas.draw()
    def remove(self):
        self.button.figure.canvas.mpl_disconnect(self.cidpress)
        self.button.figure.canvas.mpl_disconnect(self.cidmove)
        self.button.remove()
        self.text.remove()

class RadioButton:
    def __init__(self, xy, gameboard, button_names, button_funcs, width=0.8, height=0.6, dy=-0.8, dx=0):
        self.gameboard, self.buttons = gameboard, []
        for i in range(len(button_names)):
            self.buttons.append(Button((xy[0]+i*dx, xy[1]+i*dy), gameboard, button_names[i], button_funcs[i], width, height, True))

class hexTile:
    def __init__(self, x=0, y=0, perturbation=0):
        self.rs = perturbation * (np.pi/180) if type(perturbation) is int else perturbation # if perturbation is integer, assume degree and convert to radians.
        xoffset = np.array([np.sqrt(3)*x*np.cos(np.pi/6 + self.rs), np.sqrt(3)*x*np.sin(np.pi/6 + self.rs)]) # First move in x-direction
        self.center = np.array([xoffset[0]+np.sqrt(3)*y*np.cos(5*np.pi/6 + self.rs), xoffset[1]+np.sqrt(3)*y*np.sin(5*np.pi/6 + self.rs)]) # Then move in y-direction to find center
        thetas = np.arange(6)*(np.pi/3) + self.rs
        self.vertices = np.array([[self.center[0]+np.cos(t), self.center[1]+np.sin(t)] for t in thetas])
        self.patch = Polygon(self.vertices, linewidth=0.5, fc='none', ec='gray', alpha=0.5, zorder=0)
        self.x, self.y = x, y
        self.resource, self.number, self.harbor, self.gameboard, self.extraIsland = None, None, None, None, False
        self.harborFaces, self.facedByHarbor, self.clusterLabel, self.polygon, self.paintedPatches = None, set(), None, None, []
    def __eq__(self, other):
        if isinstance(other, hexTile): return (self.x, self.y) == (other.x, other.y)
        return False
    def __hash__(self):
        return hash((self.x, self.y))
    def getNeighborAttributes(self, attr='resource'):
        deltaxys = [[1,0], [1,1], [0,1], [-1,0], [-1,-1], [0,-1]]
        attrs = []
        for dx, dy in deltaxys:
            x, y = self.x+dx, self.y+dy
            if (x > self.gameboard.gs) or (x < -self.gameboard.gs) or (y > self.gameboard.gs) or (y < -self.gameboard.gs):
                continue
            attrs.append(getattr(self.gameboard[x,y], attr))
        return np.array(attrs)
    def getNeighborCoords(self):
        return set([(H.x, H.y) for H in self.neighbors])
    def getCommonNeighbors(self, neighbor):
        return set(self.neighbors).intersection(neighbor.neighbors)
    def connect(self, gameboard):
        self.gameboard = gameboard
        self.coordText = self.gameboard.ax.text(self.center[0], self.center[1], f"({self.x},{self.y})", ha='center',va='center', fontsize=10, visible=False)
        xNeighbors = self.getNeighborAttributes('x')
        yNeighbors = self.getNeighborAttributes('y')
        if len(xNeighbors) < 6: 
            self.set_resource('edge', False)
            self.isEdge = True
        else: self.isEdge = False
        self.gameboard.Neighbors[self.x,self.y] = [xNeighbors, yNeighbors]
        self.neighbors = self.gameboard[xNeighbors, yNeighbors]
        self.cidpress = self.patch.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidmotion = self.patch.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
    def removePaintedResource(self):
        if len(self.paintedPatches) > 0:
            while len(self.paintedPatches) > 0:
                paintedPatch = self.paintedPatches.pop()
                paintedPatch.remove()
            gc.collect()
    def addPixelPaint(self, pixelPaint):
        self.paintedPatches.append(pixelPaint)
        self.gameboard.ax.add_patch(pixelPaint)
    def paintResource(self, tol=1e-4):
        if self.polygon is None:
            self.polygon = Poly(self.vertices)
        m, M = self.gameboard.Mins[self.x,self.y], self.gameboard.Maxs[self.x,self.y]
        xrange = np.linspace(m[0]+tol, M[0]-tol, self.gameboard.paintedPixels+1)
        yrange = np.linspace(m[1]+tol, M[1]-tol, self.gameboard.paintedPixels+1)
        if len(self.paintedPatches) > 0: self.removePaintedResource()
        self.patch.set_fc('none')
        self.patch.set_ec('none')
        img = self.gameboard.resImages[self.resource]
        for i in range(len(xrange)-1):
            for j in range(len(yrange)-1):
                b = box(xrange[i], yrange[j], xrange[i+1], yrange[j+1])
                I = self.polygon.intersection(b)
                if I.area > 0:
                    color = '#%02x%02x%02x' % img[i, j] # Image Indexing Transform: [j, len(xrange)-2-i], although it looks like PIL accounts for it.
                    self.addPixelPaint(PolygonPatch(I, fc=color, ec=color, lw=0.05, zorder=1))
    def set_patchParams(self, draw=True, **kwargs):
        options = {'fc':self.patch.set_fc, 'ec':self.patch.set_ec, 'alpha':self.patch.set_alpha, 'num':self.set_number}
        for key, value in kwargs.items():
            if key not in options: raise KeyError(f"Unrecognized Argument: {key}")
            options[key](value)
        if draw: self.patch.figure.canvas.draw()
    def set_resource(self, resource, draw=True, hide=False):
        if resource is None: self.extraIsland = False
        self.gameboard.Resources[self.x, self.y] = resource
        self.resource = resource
        if resource not in hexColors: raise KeyError(f"Unrecognized Resource: {resource}")
        if hide and ('num' in hexColors[resource]):
            self.set_number(hexColors[resource]['num'])
        else:
            self.set_patchParams(draw, **hexColors[resource])
        if hide: self.coordText.set_visible(True)
    def set_number(self, number, hide=False):
        self.gameboard.Numbers[self.x, self.y] = number
        self.number = number
        if number not in {None, 0}:
            self.gameboard.numAffinityByRes[self.resource] += numAffinity[number]
            self.token = Circle(self.center, 0.3, fc='bisque', ec='papayawhip', lw=0.5, alpha=0.8, visible=not hide, zorder=2)
            self.gameboard.ax.add_patch(self.token)
            color = 'r' if number in {6, 8} else 'k'
            self.tokenNumber = self.gameboard.ax.text(self.center[0], self.center[1], number, color=color, va='center', ha='center', fontsize=self.gameboard.tTS, visible=not hide, zorder=3)
    def remove_number(self):
        self.number = None
        self.gameboard.Numbers[self.x, self.y] = None
        if hasattr(self, 'token'): 
            self.token.remove()
            delattr(self, 'token')
        if hasattr(self, 'tokenNumber'): 
            self.tokenNumber.remove()
            delattr(self, 'tokenNumber')
    def set_harbor(self, harbor, neighbor, hide=False):
        self.harbor = harbor
        self.harborFaces = neighbor
        neighbor.facedByHarbor.add(self)
        harborCenter = np.mean([self.center, neighbor.center], 0)
        numerator, denominator = neighbor.center[1] - self.center[1], neighbor.center[0] - self.center[0]
        if denominator == 0:
            thetaStart = np.pi if numerator < 0 else 0
        elif denominator < 0:
            thetaStart = np.tanh(numerator / denominator) + np.pi/2 # because we have to add by 180 degrees when adjacent (denominator) is negative, plus pi/2 for perpendicular
        else:
            thetaStart = np.tanh(numerator / denominator) - np.pi/2 # perpendicular of tan^-1(o/a)
        self.harborPatch = harborPatch(harborCenter, thetaStart, fc='bisque', ec='papayawhip', alpha=0.8, visible=not hide, zorder=2)
        self.gameboard.ax.add_patch(self.harborPatch)
        self.harborPatchText = self.gameboard.ax.text(harborCenter[0], harborCenter[1], harbor, va='bottom', ha='center', fontsize=self.gameboard.hTS, visible=not hide,
                                                      rotation=(180/np.pi)*thetaStart, rotation_mode='anchor', zorder=3)
        self.gameboard.TilesWithHarbor.append(self)
    def pick_fromDF(self, DF, attr):
        pdf = np.zeros((len(DF),),dtype=int)
        for H in self.neighbors: pdf += DF[getattr(H, attr)] # gets pdf
        if attr == 'number': 
            if self.resource != 'gold':
                pdf += self.gameboard.numAffinityByRes[self.resource] # Also take into account how prevelent the resource is
            else:
                pdf += numAffinity['gold']
        shiftedpdf = pdf + (1 - np.min(pdf))
        cdf = shiftedpdf.cumsum()
        if self.gameboard.negForbids and not np.all(pdf < 0): cdf[pdf < 0] = 0 # This means negative pdf value forbids affinity -- this is currently broken!
        choice = rand_from_cdf(cdf)
        if ((attr=='number') and (not self.gameboard.numRestrict)) or ((attr=='resource') and (not self.gameboard.resRestrict)): return choice
        if attr=='number':
            while self.gameboard.allowedNum[choice] <= 0: choice = rand_from_cdf(cdf)
            self.gameboard.allowedNum[choice] -= 1
        else:
            while self.gameboard.allowedRes[choice] <= 0: choice = rand_from_cdf(cdf)
            self.gameboard.allowedRes[choice] -= 1
        return choice
    def pick_resource(self, hide=False):
        self.removePaintedResource()
        self.set_resource(self.pick_fromDF(resAffinity, 'resource'), False, hide)
    def pick_resource_byCluster(self, cluster_i, hide=False):
        if (self.gameboard.clusterLandRem[cluster_i] == 0) or (self.clusterLabel != cluster_i) or (self not in self.gameboard.gameTilesRemaining): return
        self.removePaintedResource()
        neighbors = list(self.neighbors)
        baseProbability = dc(baseProbs)
        for neighbor in neighbors:
            if neighbor.resource in landTilesSet:
                baseProbability[landTilesSet[neighbor.resource]] = baseProbability[landTilesSet[neighbor.resource]] ** 2
        if self.gameboard.resRestrict:
            for resource, value in self.gameboard.allowedRes.items():
                if (value <= 0) and (resource != 'water'):
                    baseProbability[landTilesSet[resource]] = 0
        if np.sum(baseProbability) == 0: return
        resource = landTiles[rand_from_pdf(baseProbability)]
        if self.gameboard.resRestrict:
            self.gameboard.allowedRes[resource] -= 1
        self.set_resource(resource, False, hide)
    def pick_number(self, hide=False):
        self.set_number(self.pick_fromDF(numAffinity, 'number'), hide)
    def pick_number_byPDF(self, hide=False):
        pdf = dc(numProbs)
        for neighbor in self.neighbors:
            if (neighbor.number is None) or (neighbor.number==0): continue
            rank = numWeight[neighbor.number]
            if rank != 3:
                # These calculations cancel eath other out in the case when number is 4 or 10.
                pdf[rank-1] **= 2
                pdf[-rank] **= 2
                opposite_rank = 6 - rank
                pdf[opposite_rank-1] **= 0.5
                pdf[-opposite_rank] **= 0.5
            pdf[numTokensSet[neighbor.number]] **= 2
        if self.gameboard.numRestrict:
            for number, value in self.gameboard.allowedNum.items():
                if value <= 0: pdf[numTokensSet[number]] = 0
        if np.sum(pdf) == 0: pdf = dc(numProbs)
        number = numTokens[rand_from_pdf(pdf)]
        if self.gameboard.numRestrict: self.gameboard.allowedNum[number] -= 1
        self.set_number(number, hide)
    def reveal(self, draw=True):
        if self.gameboard.hiddenActivated and (self not in self.gameboard.hiddenRevealed) and (self.resource not in {'border','outside'}):
            self.gameboard.hiddenRevealed.add(self)
            self.coordText.set_visible(False)
            self.set_patchParams(False, **hexColors[self.resource])
            if hasattr(self, 'token'):
                self.token.set_visible(True)
                self.tokenNumber.set_visible(True)
            if hasattr(self, 'harborPatch'):
                vis=True if (self.harborFaces.resource == 'border') or (self.harborFaces in self.gameboard.hiddenRevealed) else False
                self.harborPatch.set_visible(vis)
                self.harborPatchText.set_visible(vis)
            for H in self.facedByHarbor:
                if H in self.gameboard.hiddenRevealed:
                    H.harborPatch.set_visible(True)
                    H.harborPatchText.set_visible(True)
            if draw: self.patch.figure.canvas.draw()
    def on_press(self, event):
        if event.inaxes != self.patch.axes: return
        contains, attrd = self.patch.contains(event)
        if not contains: return
        if not self.gameboard.isClosed:
            if type(self.resource) is not type(None):
                self.set_resource('edge') if self.isEdge else self.set_resource(None)
            else:
                self.set_resource('soft border')
            self.gameboard.update_isClosed()
            self.gameboard.activateMotion(self.x, self.y)
        self.reveal()
    def on_motion(self, event):
        if not self.gameboard.motionActivated: return
        if self.gameboard.motionTouched[self.x, self.y]: return
        self.on_press(event)
        
class Catan:
    def __init__(self, gridsize=10, gameMode='hidden', resAlgorithm='clusterMethod', numberRestriction=True,
                 perturbation=0, figsize=(12,10), clipEdges=True, paintedPixels=100, negativeForbids=False,
                 tokenTextSize=14, harborTextSize=8):
        self.gs, self.gp = gridsize, perturbation # store parameters
        self.negForbids = negativeForbids
        self.matrixSize = (gridsize*2+1, gridsize*2+1)
        self.gameMode, self.resAlgorithm, self.resRestrict, self.numRestrict = gameMode, resAlgorithm, None, numberRestriction
        self.grid = np.empty(self.matrixSize, dtype=object) # Initialize empty gridspace
        self.Maxs, self.Mins = np.zeros((gridsize*2+1,gridsize*2+1,2), dtype=float), np.zeros((gridsize*2+1,gridsize*2+1,2), dtype=float)
        self.range = np.arange(-gridsize, gridsize+1, dtype=int)
        for x in self.range:
            for y in self.range:
                self.grid[x,y] = hexTile(x, y, perturbation)
                self.Maxs[x,y] = np.max(self[x,y].vertices,0)
                self.Mins[x,y] = np.min(self[x,y].vertices,0)
        self.paintedPixels = paintedPixels
        self.isClosed = False
        self.motionActivated, self.motionTouched = False, np.zeros(self.matrixSize, dtype=bool)
        self.hiddenActivated, self.hiddenRevealed, self.unRevealedContinents = False, set(), set()
        self.buttons_assigned, self.radioButtons_assigned = False, False
        self.gameTilesRemaining = set()
        self.Resources = np.empty(self.matrixSize, dtype=object)
        self.Numbers = np.empty(self.matrixSize, dtype=object)
        self.TilesWithHarbor = []
        self.Neighbors = np.empty(self.matrixSize, dtype=object)
        self.tTS, self.hTS = tokenTextSize, harborTextSize
        self.currentStage = 'Define Borders' # goes in order of 'Define Borders', 'Borders Set', 'Resources Set', 'Numbers Set', 'Harbors Set'
        self.displayGrid(figsize, clipEdges)
    def __getitem__(self, xy):
        return self.grid[xy]
    def get_waterRatio(self):
        totalLand = len(self[self.numberableTiles])
        totalWater = np.sum(self.Resources=='water')
        total = totalLand + totalWater
        if total == 0: return 0
        return totalWater / total
    def set_resRestrict(self, resourceRestriction):
        self.resRestrict = resourceRestriction
        self.setResources(self.hiddenActivated)
        if (self.gameMode == 'complete') or (self.gameMode == 'hidden'):
            self.setNumbers(self.hiddenActivated)
            self.setHarbors(self.hiddenActivated)
        self.fig.canvas.draw()
    def setupResourceLimit(self, total):
        if self.resRestrict:
            if self.resRestrict not in resRatios: raise AssertionError(f'The resource restriction parameter "{self.resRestrict}" is not understood.')
            unitResources, allowedResources = [], []
            for resource, amount in resRatios[self.resRestrict].items():
                unitResources += [resource]*amount
                allowedResources.append(resource)
            self.allowedRes = sample2total(unitResources, total, allowedResources)
    def setupNumberLimit(self, total):
        if self.numRestrict: self.allowedNum = sample2total([2,3,3,4,4,5,5,6,6,8,8,9,9,10,10,11,11,12], total, False)
    def activateMotion(self, x, y):
        if not self.isClosed: 
            self.motionActivated = True
            self.motionTouched[x, y] = True
    def deactivateMotion(self, event):
        if self.motionActivated:
            self.motionActivated = False
            self.motionTouched[:] = False
    def zoomGrid(self, fitResource=False):
        if fitResource or (type(fitResource) is type(None)):
            mins = np.min(self.Mins[self.Resources==fitResource], 0)
            maxs = np.max(self.Maxs[self.Resources==fitResource], 0)
            self.ax.set_xlim([mins[0], maxs[0]])
            self.ax.set_ylim([mins[1], maxs[1]])
        else:
            bb = self.gs #np.sqrt(3)*(self.gs-1)
            self.ax.set_xlim([-bb, bb])
            self.ax.set_ylim([-bb, bb])
        self.fig.tight_layout()
        self.fig.canvas.draw()
    def displayGrid(self, figsize=(12,10), clipEdges=True):
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        for x in self.range:
            for y in self.range:
                self.ax.add_patch(self[x,y].patch)
                self[x,y].connect(self)
        self.fig.canvas.mpl_connect('button_release_event', self.deactivateMotion)
        fitToEdge = 'edge' if not clipEdges else False
        self.zoomGrid(fitToEdge)
    def assign_resButtons(self, width=0.9, height=0.6):
        borders = self[self.Resources=='border']
        xmin = np.min([border.center[0] for border in borders]) - width/2
        ymax = np.max([border.center[1] for border in borders])
        restrictions = [key for key in resRatios] + ['None']
        def set_funcs(key):
            def commit_funct():
                self.set_resRestrict(key)
            return commit_funct
        functions = [set_funcs(key) for key in resRatios] + [set_funcs(False)]
        self.resButtons = RadioButton((xmin,ymax), self, restrictions, functions)
    def close_resButtons(self):
        if hasattr(self, 'resButtons'):
            for button in self.resButtons.buttons: button.remove()
            delattr(self, 'resButtons')
    def assign_buttons(self, width=0.9, height=0.6):
        self.buttons_assigned = True
        borders = self[self.Resources=='border']
        xmax = np.max([border.center[0] for border in borders])
        ymax = np.max([border.center[1] for border in borders])
        reset_button = Button((xmax,ymax), self, 'Reset', self.resetBoard, width, height)
        if self.gameMode == 'hidden':
            reveal_button = Button((xmax,ymax-1.5*height), self, 'Reveal', self.revealAll, width, height)
            continent_button = Button((xmax,ymax-3*height), self, 'Continent', self.revealContinent, width, height)
            self.optionButtons = [reset_button, reveal_button, continent_button]
        else:
            redraw_button = Button((xmax,ymax-1.5*height), self, 'Redraw', self.redrawStage, width, height)
            next_button = Button((xmax,ymax-3*height), self, 'Next', self.nextStage, width, height)
            back_button = Button((xmax,ymax-4.5*height), self, 'Back', self.previousStage, width, height)
            self.optionButtons = [reset_button, redraw_button, next_button, back_button]
    def close_buttons(self):
        for button in self.optionButtons: button.remove()
        delattr(self, 'optionButtons')
    def update_isClosed(self):
        self.isClosed = False
        isBorder = self.Resources == 'soft border'
        if len(isBorder) == 0: return
        for x, y in self.Neighbors[isBorder]:
            if np.sum(self.Resources[x, y] == 'soft border') < 2: return
        self.isClosed = True
        self.deactivateMotion(None)
        for H in self[isBorder]: H.set_resource('border', False)
        self.hiddenActivated = True if self.gameMode == 'hidden' else False
        self.setBorders()
        self.assign_resButtons()
        self.fig.canvas.draw()
    def setBorders(self):
        outsideSet = list(self[self.Resources == 'edge'])
        while len(outsideSet) > 0:
            H = outsideSet.pop(0)
            H.set_resource('outside', False)
            xs, ys = self.Neighbors[H.x,H.y]
            for neighbor in self[xs, ys]:
                if neighbor.resource not in {'edge', 'border', 'outside', 'processing'}: 
                    neighbor.resource = 'processing'
                    outsideSet.append(neighbor)
        self.currentStage = 'Borders Set'
        self.zoomGrid('border')
    def clusterMethod(self, gameTiles, hide, lR=15):
        self.gameTilesRemaining = set(gameTiles)
        for H in gameTiles: H.set_resource('water', False, self.hiddenActivated)
        landClusters = int(np.round(len(gameTiles) / lR)) # use this as an approximate number of land clusters to place on board.
        minLandMasses = max([1, landClusters])
        maxLandMasses = max([1, min([max([minLandMasses, landClusters + 2]), len(gameTiles)])])
        k = np.random.randint(minLandMasses, maxLandMasses) # number of clusters randomly chosen.
        centers = np.array([H.center for H in gameTiles])
        centers_full = np.concatenate((centers, np.array([H.center for H in self[self.Resources=='border']])), 0)
        kmeans = KMeans(k, 'random', n_init=1)
        kmeans.fit(centers_full)
        clusterLabels = kmeans.predict(centers)
        for i in range(len(clusterLabels)): gameTiles[i].clusterLabel = clusterLabels[i]
        if self.resRestrict == False:
            totalLandTiles = int(len(gameTiles)*land2waterRatio)
        else:
            totalLandTiles = 0
            for landTile in landTiles: totalLandTiles += self.allowedRes[landTile]
        self.clusterLandRem, self.clusterCenters, cluster_i, TLT = {i:0 for i in range(k)}, {}, 0, totalLandTiles-k
        while totalLandTiles > 0:
            if cluster_i >= k: cluster_i = 0
            self.clusterLandRem[cluster_i] += 1
            cluster_i += 1
            totalLandTiles -= 1
        Orders = []
        for cluster_i in range(k):
            order, seen, cluster_size = [], set(), np.sum(clusterLabels == cluster_i)
            Q = [gameTiles[np.argmin(normalize(centers - kmeans.cluster_centers_[cluster_i], return_norm=True)[1])]]
            self.clusterCenters[cluster_i] = Q[0]
            order.append(Q[0])
            seen.add(Q[0])
            while len(order) < cluster_size:
                H = Q.pop(0)
                neighbors = np.random.choice(list(H.neighbors), len(H.neighbors), False)
                for nH in neighbors:
                    if nH in seen: continue
                    seen.add(nH)
                    if nH.clusterLabel != cluster_i: continue
                    order.append(nH)
                    Q.append(nH)
            Orders.append(order)
        indexes = [-1 for cluster_i in range(k)]
        while TLT > 0:
            for cluster_i in range(k):
                indexes[cluster_i] += 1
                H = Orders[cluster_i][indexes[cluster_i]]
                H.pick_resource_byCluster(cluster_i, self.hiddenActivated)
                self.gameTilesRemaining.remove(H)
                self.clusterLandRem[cluster_i] -= 1
                TLT -= 1
        # pick k random tiles to spread however!
        potentialExtras, remainder = [], []
        if self.resRestrict!=False:
            for landTile in landTiles:
                if self.allowedRes[landTile] > 0:
                    remainder += [landTile]*self.allowedRes[landTile]
            remainder = list(np.random.choice(remainder, len(remainder), False))
        for H in self.gameTilesRemaining:
            x, y = self.Neighbors[H.x, H.y]
            neighborResources = self.Resources[x, y]
            nonLand = np.sum((neighborResources=='water')+(neighborResources=='border'))
            if nonLand >= 6:
                potentialExtras.append(H)
                H.extraIsland = True
                if (len(potentialExtras) >= k) and (self.resRestrict==False): break
        for H in potentialExtras:
            if self.resRestrict==False:
                H.pick_resource_byCluster(H.clusterLabel, self.hiddenActivated)
            else:
                if len(remainder) == 0: break
                H.set_resource(remainder.pop(), False, self.hiddenActivated)
        self.gameTilesRemaining = set()
        self.continents = k
        self.unRevealedContinents = list(np.random.choice(range(k), k, False))
    def affinityMethod(self, gameTiles, hide):
        self.gameTilesRemaining = set(gameTiles)
        for H in np.random.choice(gameTiles, len(gameTiles), False): 
            H.pick_resource(hide)
            self.gameTilesRemaining.remove(H)
    def setResources(self, hide=False):
        if self.currentStage not in {'Resources Set', 'Borders Set'}: raise AssertionError("Resources cannot be set in current state")
        self.close_resButtons()
        if not self.buttons_assigned: self.assign_buttons()
        self.numAffinityByRes = pd.DataFrame(np.zeros((len(numAffinity),6),dtype=int), columns=resAffinity.columns[4:], index=numAffinity.index)
        self.gameTiles = (self.Resources!='outside')*(self.Resources!='border')
        gameTiles = self[self.gameTiles]
        self.setupResourceLimit(len(gameTiles))
        if self.resAlgorithm == 'affinityMatrix':
            self.affinityMethod(gameTiles, hide)
        else:
            self.clusterMethod(gameTiles, hide)
        #for H in np.random.choice(gameTiles, len(gameTiles), False): H.pick_resource(hide) # matrixAffinity algorithm
        self.currentStage = 'Resources Set'
    def removeResources(self):
        if self.currentStage != 'Resources Set': raise AssertionError("Resources cannot be removed in current state")
        for H in self[self.gameTiles]: H.set_resource(None, False)
        self.unRevealedContinents = set()
        self.close_buttons()
        self.buttons_assigned = False
        self.assign_resButtons()
        self.currentStage = 'Borders Set'
    def setNumbers(self, hide=False):
        if self.currentStage != 'Resources Set': raise AssertionError("Numbers cannot be set in current state")
        self.numberableTiles = self.gameTiles * (self.Numbers==None)
        numberableTiles = self[self.numberableTiles]
        self.setupNumberLimit(len(numberableTiles))
        if self.resAlgorithm == 'clusterMethod': 
            for H in np.random.choice(numberableTiles, len(numberableTiles), False): H.pick_number_byPDF(hide)
        else:
            for H in np.random.choice(numberableTiles, len(numberableTiles), False): H.pick_number(hide)
        self.currentStage = 'Numbers Set'
    def removeNumbers(self):
        if self.currentStage != 'Numbers Set': raise AssertionError("Numbers cannot be removed in current state")
        numberableTiles = self[self.numberableTiles]
        for H in numberableTiles: H.remove_number()
        self.hiddenRevealed = set()
        self.currentStage = 'Resources Set'
    def setHarbors(self, hide=False):
        if self.currentStage != 'Numbers Set': raise AssertionError("Harbors cannot be set in current state")
        Hs = self[self.numberableTiles]
        totalHarbors = calcTotalHarbors(len(Hs))
        randomHarbors = sample2total(harborRatio, totalHarbors)
        sortedHs, i = sorted(Hs, key=lambda x: -np.abs(x.number-7)), 0
        while (len(randomHarbors) > 0) and (i < len(sortedHs)):
            cH = sortedHs[i]
            for neighbor in cH.neighbors:
                if neighbor.resource in {'border', 'water'}:
                    commonNeighbors = cH.getCommonNeighbors(neighbor)
                    if np.sum([cN.harborFaces==neighbor for cN in commonNeighbors]) == 0:
                        cH.set_harbor(randomHarbors.pop(), neighbor, hide)
                        break
            i += 1
        self.currentStage = 'Harbors Set'
    def removeHarbors(self):
        if self.currentStage != 'Harbors Set': raise AssertionError("Harbors cannot be removed in current state")
        while len(self.TilesWithHarbor) > 0:
            H = self.TilesWithHarbor.pop()
            H.harborPatch.remove()
            H.harborPatchText.remove()
            H.harborFaces.facedByHarbor = set()
            H.harbor, H.harborFaces, H.facedByHarbor = None, None, set()
            delattr(H, 'harborPatch')
            delattr(H, 'harborPatchText')
        self.currentStage = 'Numbers Set'
    def resetBoard(self):
        self.removeHarbors()
        self.removeNumbers()
        self.removeResources()
        self.setResources(self.hiddenActivated)
        self.setNumbers(self.hiddenActivated)
        self.setHarbors(self.hiddenActivated)
        self.fig.canvas.draw()
    def revealAll(self, draw=True):
        if self.currentStage != 'Harbors Set': raise AssertionError("Gameboard appears incomplete")
        for H in self[self.gameTiles]: H.reveal(draw=False)
        if draw: self.fig.canvas.draw()
    def revealContinent(self, continent=None, draw=True):
        if (self.hiddenActivated == False) or (len(self.unRevealedContinents) == 0): return
        if continent is None: continent = self.unRevealedContinents.pop()
        for H in self[self.gameTiles]:
            if (H.clusterLabel==continent) and (not H.extraIsland): H.reveal(draw=False)
        if draw: self.fig.canvas.draw()
    def nextStage(self, draw=True):
        if self.currentStage == 'Define Borders':
            print("Define borders before continuing!")
            return
        if self.currentStage == 'Harbors Set':
            print("Gameboard is complete. No stages left.")
            return
        nextByStage = {'Borders Set':self.setResources, 'Resources Set':self.setNumbers, 'Numbers Set':self.setHarbors}
        nextByStage[self.currentStage]()
        if draw: self.fig.canvas.draw()
    def previousStage(self, ignoreResourcesSet=False, draw=True):
        if self.currentStage in {'Define Borders', 'Borders Set'}:
            print("Please restart script if you wish to redefine borders")
            return
        if self.currentStage == 'Harbors Set':
            self.removeHarbors()
        elif self.currentStage == 'Numbers Set':
            self.removeNumbers()
        elif self.currentStage == 'Resources Set':
            if ignoreResourcesSet:
                self.currentStage = 'Borders Set'
            else:
                self.removeResources()
        if draw: self.fig.canvas.draw()
    def redrawStage(self, draw=True):
        self.previousStage(True, False)
        self.nextStage(draw)
    def paintBoard(self, resFiles=resFiles):
        start = time.time()
        self.resImages = {}
        for resource, imageFile in resFiles.items():
            PILimg = Image.open(imageFile)
            zoomedPILimg = PILimg.resize((self.paintedPixels, self.paintedPixels))
            self.resImages[resource] = zoomedPILimg.load()
        for H in self[self.gameTiles]: H.paintResource()
        print(f"Took {(time.time()-start)/60} minutes to paint board")
     
c = Catan()