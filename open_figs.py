# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import gym
from gym import spaces
import cv2
#from google.colab.patches import cv2_imshow
#from google.colab import output
#import time
import os, sys
#os.environ["SDL_VIDEODRIVER"] = "dummy"
import matplotlib.pyplot as plt
#plt.rcParams["figure.dpi"] = 300
from matplotlib import colors
#import networkx as nx
#from networkx.drawing.nx_agraph import write_dot
#from networkx.drawing.nx_pydot import write_dot

#import pygame
#from sklearn.preprocessing import normalize
#import graphviz
#from graphviz import Source
#np.set_printoptions(threshold=sys.maxsize)
import pickle
np.set_printoptions(edgeitems=30, linewidth=100000,
    formatter=dict(float=lambda x: "%.3g" % x))
#import ffmpeg
#import moviepy.video.io.ImageSequenceClip
np.set_printoptions(precision=4)
from agents import Q_transmit_agent
from agents import AC_Agent
from env import transmit_env
from visualize import render
draw = render()

figx = pickle.load(open('FigureObject_3.fig.pickle', 'rb'))
ggg = figx.figure
dummy = plt.figure()
new_manager = dummy.canvas.manager
new_manager.canvas.figure = figx
figx.set_canvas(new_manager.canvas)
plt.show() # Show the figure, edit it, etc.!
