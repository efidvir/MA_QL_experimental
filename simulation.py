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

#agent type
#agent_type = 'Actor-Critic'
agent_type = 'Q_Learning'
#figx = pickle.load(open('FigureObject.fig.pickle', 'rb'))
#ggg = figx.figure
#ggg.show() # Show the figure, edit it, etc.!
#Global parameters
number_of_iterations = 1000000
force_policy_flag = True
number_of_agents = 3
np.random.seed(0)
mat = []
#model
MAX_SILENT_TIME = 6
SILENT_THRESHOLD = 0
BATTERY_SIZE = 6
DISCHARGE = 2
MINIMAL_CHARGE = 2
CHARGE = 1
number_of_actions = 2

#learning params
GAMMA = 0.9
ALPHA = 0.1
alphas = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
P_s = [0.1,0.5 ,0.9]
#P_LOSS = 0
decay_rate = 0.99999

#for rendering
DATA_SIZE = 10
av=2
#surface_mat = np.empty([len(alphas), number_of_iterations])
surface_mat0 = np.empty([len(alphas), int(number_of_iterations/1000)])
surface_mat1 = np.empty([len(alphas), int(number_of_iterations/1000)])
surface_mat2 = np.empty([len(alphas), int(number_of_iterations/1000)])
for p in P_s:
    for var in range(av):
        for ALPHA in alphas:
            #MAX_SILENT_TIME = sp
            #BATTERY_SIZE = sp
            '''run realtime experiences'''
            #T = [[] for i in range(number_of_agents)]
            #for i in range(number_of_agents):
            #    T[i] = np.zeros(shape=(BATTERY_SIZE * MAX_SILENT_TIME, MAX_SILENT_TIME * BATTERY_SIZE))  # transition matrix
            policies = [[] for i in range(number_of_agents)]
            values = [[] for i in range(number_of_agents)]
            #pol_t = np.ndarray(shape=(number_of_iterations, number_of_agents, BATTERY_SIZE, MAX_SILENT_TIME))
            #val_t = np.ndarray(shape=(number_of_iterations, number_of_agents, BATTERY_SIZE, MAX_SILENT_TIME))

            occupied = 0
            epsilon = np.ones(number_of_agents)
            print(epsilon)
            # initialize environment
            env = [[] for i in range(number_of_agents)]
            agent = [[] for i in range(number_of_agents)]
            state = [[] for i in range(number_of_agents)]
            actions = [[] for i in range(number_of_agents)]
            transmit_or_wait_s = [[] for i in range(number_of_agents)]
            score = [[] for i in range(number_of_agents)]
            RAND = [[np.random.randint(10000)] for i in range(number_of_agents)]
            rewards = [[] for i in range(number_of_agents)]
            avg_rwrd  = [[] for i in range(number_of_agents)]
            for i in range(number_of_agents):
                #epsilon[i] = epsilon[i] -1/(number_of_agents+i)
                env[i] = transmit_env(BATTERY_SIZE, MAX_SILENT_TIME, SILENT_THRESHOLD, MINIMAL_CHARGE, DISCHARGE, CHARGE, DATA_SIZE, number_of_actions)
                if agent_type == 'Q_Learning':
                    agent[i] = Q_transmit_agent(ALPHA, GAMMA, BATTERY_SIZE, MAX_SILENT_TIME, DATA_SIZE, number_of_actions, MINIMAL_CHARGE,RAND[i])
                    #Q_tables = [[] for i in range(number_of_iterations)]
                elif agent_type == 'Actor-Critic':
                    agent[i] = AC_Agent(5*i*0.0000008, GAMMA, BATTERY_SIZE, MAX_SILENT_TIME, DATA_SIZE, number_of_actions,MINIMAL_CHARGE)
                    print('Make sure to adjust the learning rate')
                state[i] = env[i].initial_state
                actions[i] , transmit_or_wait_s[i] = agent[i].choose_action(state[i], epsilon[i],p)
                #policies[i] = agent[i].get_policy()
                #values[i] = agent[i].get_state_value(policies[i])


            # plot reward function in use
            #plt.plot(range(len(env[0].r_1)), env[0].r_1, 'o--', color='blue')
            #plt.xticks(range(env[0].max_silence_time))
            #plt.title('Reward function $r_1$')
            #plt.show(block=False)
            #print(epsilon)
            #print('r_1 array: ', env[0].r_1)

            errors = [[] for i in range(number_of_agents)]

            for i in range(number_of_iterations):

                # all agents move a step and take a new action
                for j in range(number_of_agents):
                    env[j].state = env[j].new_state
                # Gateway decision
                if sum(transmit_or_wait_s) > 1 or sum(transmit_or_wait_s) == 0:
                    ack = 0
                elif sum(transmit_or_wait_s) == 1:
                    ack = 1

                for j in range(number_of_agents):
                    new_state, reward, occupied = env[j].time_step(actions[j], transmit_or_wait_s[j], sum(transmit_or_wait_s), ack)  # CHANNEL
                    rewards[j] = reward

                    env[j].new_state = new_state
                    score[j].append(reward)

                for j in range(number_of_agents):
                    np.random.seed(j)
                    #print('Agent ', j)
                    #draw.render_Q_diffs(agent[j].Q[:, :, 0], agent[j].Q[:, :, 1], j,i,env[j].state,actions[j], rewards[j], env[j].new_state)
                    actions[j], transmit_or_wait_s[j] = agent[j].step(env[j].state, rewards[j], actions[j], transmit_or_wait_s[j], env[j].new_state, epsilon[j],p)
                    epsilon[j] = epsilon[j] * decay_rate
                    #agent[j].alpha = agent[j].alpha * decay_rate

                if i % 100 == 0:
                    print(p, var, ALPHA,'step: ', i, '100 steps AVG mean score: ',np.mean(score[0][-100:-1]),epsilon[0])

            for i in range(int(len(score[0])/1000)):
                avg_rwrd[0].append(np.mean(score[0][1000*i:1000*(i+1)]))

            if mat==[]:
                mat = np.array(avg_rwrd[0])
                max_mat = np.array(avg_rwrd[0])
                min_mat = np.array(avg_rwrd[0])
                avg_mat = np.array(avg_rwrd[0])
            else:
                #print(mat.shape(),avg_rwrd.shape())
                mat = np.vstack((mat,np.array(avg_rwrd[0])))
        #plt.plot(range(len(avg_rwrd[0])), avg_rwrd[0]):
    #plt.plot(range(len(np.sum(mat, axis = 0))), np.sum(mat, axis = 0))
    mat = np.delete(mat, 0, axis=0)
    for al in range(len(alphas)):
        max_mat = np.vstack((max_mat,np.amax(mat[al::av], axis=0)))
        min_mat = np.vstack((min_mat,np.amin(mat[al::av], axis=0)))
        avg_mat = np.vstack((avg_mat,np.average(mat[al::av], axis=0)))
    max_mat = np.delete(max_mat, 0, axis=0)
    min_mat = np.delete(min_mat, 0, axis=0)
    avg_mat = np.delete(avg_mat, 0, axis=0)
    #plt.plot(range(len(avg_rwrd[0])), avg_mat[0])
    #plt.fill_between(range(len(avg_rwrd[0])), min_mat[0], max_mat[0], facecolor='blue', alpha=0.5, label='1 sigma range')
    #plt.plot(range(len(avg_rwrd[0])), avg_mat[1])
    #plt.fill_between(range(len(avg_rwrd[0])), min_mat[1], max_mat[1], facecolor='orange', alpha=0.5, label='1 sigma range')

    #plt.plot(range(len(avg_rwrd[0])), avg_mat[2])
    #plt.fill_between(range(len(avg_rwrd[0])), min_mat[2], max_mat[2], facecolor='green', alpha=0.5, label='1 sigma range')
    if P_s.index(p) == 0:
        surface_mat0 = np.vstack((avg_mat[0],avg_mat[1],avg_mat[2],avg_mat[3],avg_mat[4],avg_mat[5]))
        #surface_mat0 = np.vstack((surface_mat0,avg_mat[2]))
    if P_s.index(p) == 1:
        surface_mat1 = np.vstack((avg_mat[0],avg_mat[1],avg_mat[2],avg_mat[3],avg_mat[4],avg_mat[5]))
        #surface_mat1 = np.vstack((surface_mat0,avg_mat[2]))
    if P_s.index(p) == 2:
        surface_mat2 = np.vstack((avg_mat[0],avg_mat[1],avg_mat[2],avg_mat[3],avg_mat[4],avg_mat[5]))
        #surface_mat2 = np.vstack((surface_mat0,avg_mat[2]))

X,Y = np.meshgrid(range(int(number_of_iterations/1000)),alphas)
fig, ax = plt.subplots()
ax = plt.axes(projection='3d')
p0 = ax.plot_surface(X, Y, surface_mat0, alpha=0.3, label='p = 0.1')

p1 = ax.plot_surface(X, Y, surface_mat1, alpha=0.3, label='p = 0.5')

p2 = ax.plot_surface(X, Y, surface_mat2, alpha=0.3, label='p = 0.9')

ax.set_zlabel('Average reward')
ax.set_ylabel('Learning rate')
ax.set_xlabel('Number of iterations X1000')
pickle.dump(ax, open('FigureObject.fig.pickle', 'wb'))
#ax.legend()
#plt.show()
#plt.plot(range(len(avg_rwrd[0])), avg_mat[4])
#plt.fill_between(range(len(avg_rwrd[0])), min_mat[4], max_mat[4], facecolor='green', alpha=0.5, label='1 sigma range')


    #plt.fill_between(t, lower_bound, upper_bound, facecolor='yellow', alpha=0.5, label='1 sigma range')

#plt.legend(['0.01 Average','0.01 Variance','0.5 Average','0.5 Variance','0.9 Average','0.9 Variance'])
#plt.title('Learning rates')
#plt.ylabel('Average reward')
#plt.xlabel('Number of iterations X1000')
#plt.savefig('alphas_var.pdf', dpi=1000)
#for j in range(number_of_agents):
#    draw.render_Q_diffs_video(agent[j].Q[:, :, 0], agent[j].Q[:, :, 1], j,number_of_iterations)
#    print('video done')
#df = pd.DataFrame(data, columns=gammas)
print(epsilon)
# plt.plot(errors)
#video.release()


#Agent evaluation
# No exploration
epsilon = np.zeros(number_of_agents)

data = []
collisions = 0
agent_clean = [np.zeros(1) for i in range(number_of_agents)]
wasted = 0
num_of_eval_iner = 1000

for i in range(num_of_eval_iner):
    #print(env[0].new_state,env[1].new_state,env[2].new_state)
    for a in range(number_of_agents):
        env[a].state = env[a].new_state
    # Gateway decision
    if sum(transmit_or_wait_s) > 1 or sum(transmit_or_wait_s) == 0:
        ack = 0
    elif sum(transmit_or_wait_s) == 1:
        ack = 1

    if sum(transmit_or_wait_s) > 1:
        collisions += 1
        data.append(1)
    if sum(transmit_or_wait_s) == 1:
        for a in range(number_of_agents):
            if transmit_or_wait_s[a] == 1:
                agent_clean[a] += 1
                data.append(a+2)
    if sum(transmit_or_wait_s) == 0:
        wasted += 1
        data.append(0)

    for j in range(number_of_agents):
        new_state, reward, occupied = env[j].time_step(actions[j], transmit_or_wait_s[j], sum(transmit_or_wait_s), ack)  # CHANNEL
        rewards[j] = reward

        env[j].new_state = new_state

    for j in range(number_of_agents):
        np.random.seed(j)
        #print('Agent ', j)
        #draw.render_Q_diffs(agent[j].Q[:, :, 0], agent[j].Q[:, :, 1], j,i,env[j].state,actions[j], rewards[j], env[j].new_state)
        actions[j], transmit_or_wait_s[j] = agent[j].choose_action( env[j].new_state, 0,p)
            #step(env[j].state, rewards[j], actions[j], transmit_or_wait_s[j], env[j].new_state, epsilon[j])


    #for a in range(number_of_agents):
    #    new_state, reward, occupied = env[a].time_step(actions[a],transmit_or_wait_s[a], sum(transmit_or_wait_s), ack)  # CHANNEL
    #    env[a].new_state = new_state
    #    actions[a] ,transmit_or_wait_s[a] = agent[a].choose_action(env[a].new_state, 0)#step(env[a].state, reward, actions[a],transmit_or_wait_s[a], env[a].new_state, epsilon[a])
print('collisions', collisions)
'''
for a in range(number_of_agents):
    print('agent{d}'.format(d=a), agent_clean[a] , 'rate:  ', env[a].discharge_rate)
    for i in range(int(len(score[a])/1000)):
        avg_rwrd[a].append(np.mean(score[a][1000*i:1000*(i+1)]))
    plt.plot(range(len(avg_rwrd[a])), avg_rwrd[a])
plt.legend(range(number_of_agents))
plt.show()
print('wasted', wasted)
'''
#print(data)
draw.last_1k_slots(data, number_of_agents)
'''
for i in range(number_of_agents):
    print('Agent ', i)
    print('\n')

    #draw.plot_Q_values(Q_tables,number_of_iterations)

for i in range(number_of_agents):
    print('Agent ',i,' Q table:', agent[i].Q[:, :, :])
    draw.render_Q(agent[j].Q[:, :, 0], agent[j].Q[:, :, 1], j, i, env[j].state)
    cv2.waitKey(0)
'''