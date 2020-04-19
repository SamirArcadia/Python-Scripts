# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:20:44 2020

@author: samir
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle
from PIL import Image
from shapely.geometry import box, Point
from shapely.geometry.polygon import Polygon as Poly
from descartes import PolygonPatch
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
                            [-2, -4, -6, -6,  0, -1, -1, -1, -1, -1],
                            [-2,  1,  0, -1, -9, -4, -4, -4, -4, -4],
                            [ 1,  0,  0,  0, -2, -4,  0,  0,  0,  0],
                            [ 1,  0,  0,  0, -2,  0, -4,  0,  0,  0],
                            [ 1,  0,  0,  0, -2,  0,  0, -4,  0,  0],
                            [ 1,  0,  0,  0, -2,  0,  0,  0, -4,  0],
                            [ 1,  0,  0,  0, -2,  0,  0,  0,  0, -4]],
                index=['water','desert','gold','brick','ore','wood','wheat','sheep'],
                columns=[None, 'border','water','desert','gold','brick','ore','wood','wheat','sheep'])

resFiles = {'brick':'images\\brick.jpg','desert':'images\\desert.JPG','gold':'images\\gold.JPG','ore':'images\\ore.jpg',
            'sheep':'images\\sheep.jpg','water':'images\\water.JPG','wheat':'images\\wheat.jpg','wood':'images\\wood.JPG'}

numAffinity = pd.DataFrame([[ 0,  0, -4,  1,  2,  3,  4,  4,  3,  2,  1,  0],
                            [ 1,  1,  1, -4,  1,  2,  3,  3,  2,  1,  0,  1],
                            [ 2,  2,  2,  1, -4,  1,  2,  2,  1,  0,  1,  2],
                            [ 3,  1,  3,  2,  1, -4,  1,  1,  0,  1,  2,  3],
                            [ 4,  0,  4,  3,  2,  1, -4,  0,  1,  2,  3,  4],
                            [ 4,  0,  4,  3,  2,  1,  0, -4,  1,  2,  3,  4],
                            [ 3,  1,  3,  2,  1,  0,  1,  1, -4,  1,  2,  3],
                            [ 2,  2,  2,  1,  0,  1,  2,  2,  1, -4,  1,  2],
                            [ 1,  1,  1,  0,  1,  2,  3,  3,  2,  1, -4,  1],
                            [ 0,  0,  0,  1,  2,  3,  4,  4,  3,  2,  1, -4]],
                index=[2, 3, 4, 5, 6, 8, 9, 10, 11, 12],
                columns=[0, None, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12])

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
    randomNumber = np.random.randint(1, cdf.tail(1)+1)
    chosenIndex = 0
    while randomNumber > cdf.iloc[chosenIndex]:
        chosenIndex += 1
    return cdf.index[chosenIndex]

class Button:
    def __init__(self, xy, gameboard, width=0.8, height=0.6, button_type='Reset'):
        self.gameboard, self.button_type = gameboard, button_type
        offsetx, offsety = width, height
        fcs = {'Reset':'lavender',  'Reveal':'lightyellow', 'Redraw':'lightcyan',     'Next':'lightgreen', 'Back':'antiquewhite'}
        ecs = {'Reset':'slategrey', 'Reveal':'olive',       'Redraw':'lightseagreen', 'Next':'darkgreen',  'Back':'olive'}
        tcs = {'Reset':'orangered', 'Reveal':'y',           'Redraw':'darkcyan',      'Next':'green',      'Back':'orange'}
        self.button = Rectangle(xy, offsetx, offsety, fc=fcs[button_type], ec=ecs[button_type], alpha=0.8, zorder=2)
        self.gameboard.ax.add_patch(self.button)
        self.text = self.gameboard.ax.text(xy[0]+offsetx/2, xy[1]+offsety/2, button_type, fontsize=10, color=tcs[button_type], va='center', ha='center', zorder=3)
        self.cidpress = self.button.figure.canvas.mpl_connect('button_press_event', self.on_press)
    def on_press(self, event):
        if event.inaxes != self.button.axes: return
        contains, attrd = self.button.contains(event)
        if not contains: return
        if self.button_type == 'Reset':
            self.gameboard.resetBoard()
        elif self.button_type == 'Reveal':
            self.gameboard.revealAll()
        elif self.button_type == 'Redraw':
            self.gameboard.previousStage(True, False)
            self.gameboard.nextStage()
        elif self.button_type == 'Next':
            self.gameboard.nextStage()
        elif self.button_type == 'Back':
            self.gameboard.previousStage()

class hexTile:
    def __init__(self, x=0, y=0, perturbation=0):
        self.rs = perturbation * (np.pi/180) if type(perturbation) is int else perturbation # if perturbation is integer, assume degree and convert to radians.
        xoffset = np.array([np.sqrt(3)*x*np.cos(np.pi/6 + self.rs), np.sqrt(3)*x*np.sin(np.pi/6 + self.rs)]) # First move in x-direction
        self.center = np.array([xoffset[0]+np.sqrt(3)*y*np.cos(5*np.pi/6 + self.rs), xoffset[1]+np.sqrt(3)*y*np.sin(5*np.pi/6 + self.rs)]) # Then move in y-direction to find center
        thetas = np.arange(6)*(np.pi/3) + self.rs
        self.vertices = np.array([[self.center[0]+np.cos(t), self.center[1]+np.sin(t)] for t in thetas])
        self.patch = Polygon(self.vertices, linewidth=0.5, fc='none', ec='gray', alpha=0.5, zorder=0)
        self.x, self.y = x, y
        self.resource, self.number, self.harbor, self.gameboard = None, None, None, None
        self.harborFaces, self.facedByHarbor, self.polygon, self.paintedPatches = None, set(), None, []
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
                #I = b if self.polygon.contains(b) else self.polygon.intersection(b)
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
        self.gameboard.Resources[self.x, self.y] = resource
        self.resource = resource
        if resource not in hexColors: raise KeyError(f"Unrecognized Resource: {resource}")
        if hide and ('num' in hexColors[resource]):
            self.set_number(hexColors[resource]['num'])
        else:
            self.set_patchParams(draw, **hexColors[resource])
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
        if hasattr(self, 'token'): self.token.remove()
        if hasattr(self, 'tokenNumber'): self.tokenNumber.remove()
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
        if attr == 'number': pdf += self.gameboard.numAffinityByRes[self.resource] # Also take into account how prevelent the resource is
        pdf += (1 - np.min(pdf))
        cdf = pdf.cumsum()
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
        if hide: self.coordText.set_visible(True)
    def pick_number(self, hide=False):
        self.set_number(self.pick_fromDF(numAffinity, 'number'), hide)
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
    def __init__(self, gridsize=10, gameMode='complete', resourceRestriction='standard', numberRestriction=True,
                 perturbation=0, figsize=(12,10), clipEdges=True, paintedPixels=100,
                 tokenTextSize=14, harborTextSize=8):
        self.gs, self.gp = gridsize, perturbation # store parameters
        self.matrixSize = (gridsize*2+1, gridsize*2+1)
        self.gameMode, self.resRestrict, self.numRestrict = gameMode, resourceRestriction, numberRestriction
        self.grid = np.empty(self.matrixSize, dtype=object) # Initialize empty gridspace
        self.Maxs, self.Mins = np.zeros((gridsize*2+1,gridsize*2+1,2), dtype=float), np.zeros((gridsize*2+1,gridsize*2+1,2), dtype=float)
        self.range = np.arange(-gridsize, gridsize+1, dtype=int)
        for x in self.range:
            for y in self.range:
                self.grid[x,y] = hexTile(x, y, perturbation)
                self.Maxs[x,y] = np.max(self[x,y].vertices,0)
                self.Mins[x,y] = np.min(self[x,y].vertices,0)
        self.paintedPixels = paintedPixels
        self.initiate_resImages()
        self.isClosed = False
        self.motionActivated = False
        self.motionTouched = np.zeros(self.matrixSize, dtype=bool)
        self.hiddenActivated = False
        self.hiddenRevealed = set()
        self.Resources = np.empty(self.matrixSize, dtype=object)
        self.Numbers = np.empty(self.matrixSize, dtype=object)
        self.TilesWithHarbor = []
        self.Neighbors = np.empty(self.matrixSize, dtype=object)
        self.tTS, self.hTS = tokenTextSize, harborTextSize
        self.currentStage = 'Define Borders' # goes in order of 'Define Borders', 'Borders Set', 'Resources Set', 'Numbers Set', 'Harbors Set'
        self.displayGrid(figsize, clipEdges)
    def initiate_resImages(self, resFiles=resFiles):
        self.resImages = {}
        for resource, imageFile in resFiles.items():
            PILimg = Image.open(imageFile)
            zoomedPILimg = PILimg.resize((self.paintedPixels, self.paintedPixels))
            self.resImages[resource] = zoomedPILimg.load()
    def __getitem__(self, xy):
        return self.grid[xy]
    def get_waterRatio(self):
        totalLand = len(self[self.numberableTiles])
        totalWater = np.sum(self.Resources=='water')
        total = totalLand + totalWater
        if total == 0: return 0
        return totalWater / total
    def setupResources(self, total):
        if self.resRestrict:
            if self.resRestrict not in resRatios: raise AssertionError(f'The resource restriction parameter "{self.resRestrict}" is not understood.')
            unitResources, allowedResources = [], []
            for resource, amount in resRatios[self.resRestrict].items():
                unitResources += [resource]*amount
                allowedResources.append(resource)
            self.allowedRes = sample2total(unitResources, total, allowedResources)
    def setupNumbers(self, total):
        if self.numRestrict: self.allowedNum = sample2total([2,3,4,5,6,8,9,10,11,12], total, False)
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
    def assign_buttons(self, width=0.9, height=0.6):
        borders = self[self.Resources=='border']
        xmax = np.max([border.center[0] for border in borders])
        ymax = np.max([border.center[1] for border in borders])
        self.reset_button = Button((xmax,ymax), self, width, height, 'Reset')
        if self.gameMode == 'hidden':
            self.reveal_button = Button((xmax,ymax-1.5*height), self, width, height, 'Reveal')
        else:
            self.redraw_button = Button((xmax,ymax-1.5*height), self, width, height, 'Redraw')
            self.next_button = Button((xmax,ymax-3*height), self, width, height, 'Next')
            self.back_button = Button((xmax,ymax-4.5*height), self, width, height, 'Back')
    def update_isClosed(self):
        self.isClosed = False
        isBorder = self.Resources == 'soft border'
        if len(isBorder) == 0: return
        for x, y in self.Neighbors[isBorder]:
            if np.sum(self.Resources[x, y] == 'soft border') < 2: return
        self.isClosed = True
        self.deactivateMotion(None)
        for H in self[isBorder]: H.set_resource('border', False)
        self.setBorders()
        self.hiddenActivated = True if self.gameMode == 'hidden' else False
        self.setResources(self.hiddenActivated)
        if (self.gameMode == 'complete') or (self.gameMode == 'hidden'):
            self.setNumbers(self.hiddenActivated)
            self.setHarbors(self.hiddenActivated)
        self.assign_buttons()
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
    def setResources(self, hide=False):
        if self.currentStage not in {'Resources Set', 'Borders Set'}: raise AssertionError("Resources cannot be set in current state")
        self.numAffinityByRes = pd.DataFrame(np.zeros((len(numAffinity),6),dtype=int),columns=resAffinity.columns[4:],index=numAffinity.index)
        self.gameTiles = (self.Resources!='outside')*(self.Resources!='border')
        gameTiles = self[self.gameTiles]
        self.setupResources(len(gameTiles))
        for H in np.random.choice(gameTiles, len(gameTiles), False): H.pick_resource(hide)
        self.currentStage = 'Resources Set'
    def setNumbers(self, hide=False):
        if self.currentStage != 'Resources Set': raise AssertionError("Numbers cannot be set in current state")
        self.numberableTiles = self.gameTiles * (self.Numbers==None)
        numberableTiles = self[self.numberableTiles]
        self.setupNumbers(len(numberableTiles))
        for H in np.random.choice(numberableTiles, len(numberableTiles), False): H.pick_number(hide)
        self.currentStage = 'Numbers Set'
    def removeNumbers(self):
        if self.currentStage != 'Numbers Set': raise AssertionError("Numbers cannot be removed in current state")
        numberableTiles = self[self.numberableTiles]
        for H in numberableTiles: H.remove_number()
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
            H.harbor, H.harborFaces = None, None
        self.currentStage = 'Numbers Set'
    def resetBoard(self):
        self.removeHarbors()
        self.removeNumbers()
        self.setResources()
        self.setNumbers()
        self.setHarbors()
        self.fig.canvas.draw()
    def revealAll(self):
        if self.currentStage != 'Harbors Set': raise AssertionError("Gameboard appears incomplete")
        for H in self[self.gameTiles]: H.reveal(draw=False)
        self.fig.canvas.draw()
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
        badInput = {'Define Borders', 'Borders Set'} if ignoreResourcesSet else {'Define Borders', 'Borders Set', 'Resources Set'}
        if self.currentStage in badInput:
            print("Please restart script if you wish to redefine borders")
            return
        if self.currentStage == 'Harbors Set':
            self.removeHarbors()
        elif self.currentStage == 'Numbers Set':
            self.removeNumbers()
        elif (self.currentStage == 'Resources Set') and ignoreResourcesSet:
            self.currentStage = 'Borders Set'
        if draw: self.fig.canvas.draw()
    def paintBoard(self):
        start = time.time()
        for H in self[self.gameTiles]: H.paintResource()
        print(f"Took {(time.time()-start)/60} minutes to paint board")
     
c = Catan()