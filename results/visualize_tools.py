#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Handle convergence posiibilities file, generate 3d surface plot by matplotlib.
"""

from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt
import csv

CONV_PATH = "/lockbox_ws/report/data/converge_possibilities.csv"
POINTS_PATH = "/lockbox_ws/report/data/associated_points.csv"
OBS_PATH = "/lockbox_ws/report/data/obs_articulated_error.csv"
MOD_PATH = "/lockbox_ws/report/data/mod_articulated_error.csv"
RIGID_PATH = "/lockbox_ws/report/data/rigid_error_final.csv"


# This plot shows the convergence possibilities of rigid tracking for different initial guesses
def converge_possibilities():
    Z = np.zeros((20, 20))
    with open(CONV_PATH,'r') as dest_f:
        data_iter = csv.reader(dest_f, delimiter = ',')
        data = [data for data in data_iter]
        for i in range(20):
            for j in range(20):
                Z[i][j] = float(data[i][j])


    fig = plt.figure()
    ax = plt.axes(projection="3d")
    def z_function(x, y):
        return np.sin(np.sqrt(x ** 2 + y ** 2))

    x = np.linspace(0, 47.5, 20)
    y = np.linspace(0, 57.42, 20)

    X, Y = np.meshgrid(x, y)
    # Z = z_function(X, Y)

    # ax.plot_wireframe(X, Y, Z, color='green')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='cool', edgecolor='none')
    ax.set_title('Probability of convergence after global Perturbation')
    ax.set_xlabel('Initial translation error[cm]')
    ax.set_ylabel('Initial rotation error[degrees]')
    # ax.set_zlabel('Probability')

    sm = plt.cm.ScalarMappable(cmap='cool', norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    cb = plt.colorbar(sm)
    cb.set_label('Probability of Convergence')

    plt.show()

    # plt.savefig('/home/yuchen/Documents/lockbox_ws/report/images/comv_rigid_error.png')

# This plot shows the associated points during the articulated tracking process
def associated_points():
    with open(POINTS_PATH,'r') as non_points_file:
        data_iter = csv.reader(non_points_file, delimiter = ',')
        data = [data for data in data_iter]
        non_fixed_points = np.zeros(len(data[0])-1)
        for i in range(len(data[0])-1):
            non_fixed_points[i] = int(data[0][i])

    # with open("/home/yuchen/Documents/lockbox_ws/report/images/fixed_points.csv",'r') as fixed_points_file:
    #     data_iter = csv.reader(fixed_points_file, delimiter = ',')
    #     data = [data for data in data_iter]
    #     fixed_points = np.zeros(len(data[0]))
    #     for i in range(len(data[0])):
    #         fixed_points[i] = int(data[0][i])

    fig = plt.figure()
    ax = plt.axes()
    x = np.arange(len(data[0])-2)
    ax.plot(x, non_fixed_points)
    # ax.plot(x, fixed_points[:-2], label='fixed backboard')

    ax.set_title('Number of associated points during tracking')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Associated Points')

    # plt.legend()
    plt.show()

# This plot shows the E_mod during the articulated tracking process
def err_mod():
    with open(MOD_PATH,'r') as file1:
        data_iter1 = csv.reader(file1, delimiter = ',')
        data1 = [data1 for data1 in data_iter1]
        not_fixed_mod_errors = np.zeros(len(data1[0]))
        for i in range(len(data1[0])):
            not_fixed_mod_errors[i] = float(data1[0][i])

    # with open("/home/yuchen/Documents/lockbox_ws/report/images/fixed_mod_error.csv",'r') as file2:
    #     data_iter2 = csv.reader(file2, delimiter = ',')
    #     data2 = [data2 for data2 in data_iter2]
    #     fixed_mod_errors = np.zeros(len(data2[0]))
    #     for i in range(len(data2[0])):
    #         fixed_mod_errors[i] = float(data2[0][i])

    fig = plt.figure()
    ax = plt.axes()
    x = np.arange(len(data1[0]))
    ax.plot(x, not_fixed_mod_errors, label='unfixed backboard')
    # ax.plot(x, fixed_mod_errors[:-2], label='fixed backboard')

    ax.set_title('model SDF error not fixed vs fixed')
    ax.set_xlabel('Frame')
    ax.set_ylabel('model SDF error(cm)')

    plt.legend()
    plt.show()

# This plot shows the E_mod during the articulated tracking process
def err_obs():
    with open(OBS_PATH,'r') as file1:
        data_iter1 = csv.reader(file1, delimiter = ',')
        data1 = [data1 for data1 in data_iter1]
        not_fixed_mod_errors = np.zeros(len(data1[0]))
        for i in range(len(data1[0])):
            not_fixed_mod_errors[i] = float(data1[0][i])

    # with open("/home/yuchen/Documents/lockbox_ws/report/images/fixed_obs_error.csv",'r') as file2:
    #     data_iter2 = csv.reader(file2, delimiter = ',')
    #     data2 = [data2 for data2 in data_iter2]
    #     fixed_mod_errors = np.zeros(len(data2[0]))
    #     for i in range(len(data2[0])):
    #         fixed_mod_errors[i] = float(data2[0][i])

    fig = plt.figure()
    ax = plt.axes()
    x = np.arange(len(data1[0]))
    ax.plot(x[:-3], not_fixed_mod_errors[:-3], label='mod_error')
    # ax.plot(x, fixed_mod_errors[:-2], label='fixed backboard')

    ax.set_title('observed points SDF error')
    ax.set_xlabel('Frame')
    ax.set_ylabel('obs SDF error(cm)')

    plt.legend()
    plt.show()

# This plot shows the position error during the rigid tracking process
def rigid_error():
    with open(RIGID_PATH,'r') as file1:
        data_iter1 = csv.reader(file1, delimiter = ',')
        data1 = [data1 for data1 in data_iter1]
        rigid_errors = np.zeros((len(data1), len(data1[1])))
        for i in range(len(data1)):
            for j in range(len(data1[0])):
                if data1[i][j] != ' ':
                    rigid_errors[i][j] = float(data1[i][j])*168

    fig = plt.figure()
    ax = plt.axes()
    x = np.arange(len(data1[0]))
    ini_err = np.arange(len(data1))
    ini_err = np.asarray(ini_err, dtype=float)
    for i in range(len(data1)):
        if ini_err[i] != 0.0:
            ini_err[i] = ini_err[i]*3

    colors = plt.cm.coolwarm(np.linspace(0,1,len(data1)))

    for i, line in enumerate(rigid_errors):
        plt.plot(x, line, label = 'initial error %dcm' %int(ini_err[i]), color=colors[i])

    ax.set_title('Rigid tracking with different initialization error')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Error between tracking results (position) and ground truth[cm]')
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    cb = plt.colorbar(sm)
    cb.set_label('Errors of Initial Guess')
    err_tick = rigid_errors[-1][0]
    cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb.set_ticklabels([0, round(err_tick*0.2, 1), round(err_tick*0.4, 1), 
        round(err_tick*0.6, 1), round(err_tick*0.8, 1), round(err_tick, 1)])

    # ax.legend(bbox_to_anchor=(1, 1))
    
    plt.show()

# err_obs()
# err_mod()
# converge_possibilities()
# rigid_error()
associated_points()

while True:
    pass