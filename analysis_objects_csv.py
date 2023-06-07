import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.core.fromnumeric import mean, size
import pandas as pd
from math import pi
import statistics
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

car_count = person_count = trafic_count = bus_count = truck_count = stop_count = motor_count = fire_count = bicycle_count = train_count = park_count = 0
car_total = person_total = trafic_total = bus_total = truck_total = stop_total = motor_total = fire_total = bicycle_total = train_total = park_total = np.array([])
frame_count = frame = row_count = seconds =  0
object_count = np.array([[person_count, bicycle_count, car_count, motor_count, bus_count, train_count, truck_count, trafic_count, fire_count, stop_count, park_count]])
time = []
frames = [0]

with open('objects.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if not any(row):
            continue
        row_count += 1
        if row_count != 1:
            if row[0] != frame_count: 
                frame_count = row[0]
                result = np.array([[person_count, bicycle_count, car_count, motor_count, bus_count, train_count, truck_count, trafic_count, fire_count, stop_count, park_count]])     
                object_count = np.concatenate((object_count, result), axis = 0)
                car_count = person_count = trafic_count = bus_count = truck_count = stop_count = motor_count = fire_count = bicycle_count = train_count = park_count = 0
                if int(frame_count)%30 == 0: 
                    seconds += 1
                    time.append(str(seconds))
                    frames.append(int(frame_count))

        if row[2] == "person": 
            person_count += 1
            if row[1] not in person_total:
                person_total = np.append(person_total, row[1])
        elif row[2] == "bicycle": 
            bicycle_count += 1
            if row[1] not in bicycle_total:
                bicycle_total = np.append(bicycle_total, row[1])
        elif row[2] == "car": 
            car_count += 1
            if row[1] not in car_total:
                car_total = np.append(car_total, row[1])
        elif row[2] == "motorbike": 
            motor_count += 1
            if row[1] not in motor_total:
                motor_total = np.append(motor_total, row[1])
        elif row[2] == "bus": 
            bus_count += 1
            if row[1] not in bus_total:
                bus_total = np.append(bus_total, row[1])
        elif row[2] == "train": 
            train_count += 1
            if row[1] not in train_total:
                train_total = np.append(train_total, row[1])
        elif row[2] == "truck": 
            truck_count += 1
            if row[1] not in truck_total:
                truck_total = np.append(truck_total, row[1])
        elif row[2] == "traffic light": 
            trafic_count += 1
            if row[1] not in trafic_total:
                trafic_total = np.append(trafic_total, row[1])
        elif row[2] == "fire hydrant": 
            fire_count += 1
            if row[1] not in fire_total:
                fire_total = np.append(fire_total, row[1])
        elif row[2] == "stop sign": 
            stop_count += 1
            if row[1] not in stop_total:
                stop_total = np.append(stop_total, row[1])
        elif row[2] == "parking meter": 
            park_count += 1
            if row[1] not in park_total:
                park_total = np.append(park_total, row[1])

    result = np.array([[person_count, bicycle_count, car_count, motor_count, bus_count, train_count, truck_count, trafic_count, fire_count, stop_count, park_count]])     
    object_count = np.concatenate((object_count, result), axis = 0)
    seconds += 1
    time.append(str(seconds))

# print("person, bycicle, car, motorbike, bus, train, truck, trafic light, fire hydrant, stop sign, parking meter")
# print(object_count)

# Variables
suma = np.sum(object_count, axis=0)
sum_objects = [len(person_total), len(bicycle_total), len(car_total), len(motor_total), len(bus_total), len(train_total), len(truck_total), len(trafic_total), len(fire_total), len(stop_total), len(park_total)]
names = ['person', 'bycicle', 'car', 'motorbike', 'bus', 'train', 'truck', 'trafic light', 'fire hydrant', 'stop sign', 'parking meter']
# nums = np.arange(int(frame_count))
time = np.arange(seconds)
mean_all = []

# Objects arrays
persons = object_count[:, 0]
bycicles = object_count[:, 1]
cars = object_count[:, 2]
motorbikes = object_count[:, 3]
busses = object_count[:, 4]
trains = object_count[:, 5]
trucks = object_count[:, 6]
trafics = object_count[:, 7]
fires = object_count[:, 8]
stops = object_count[:, 9]
parks = object_count[:, 10]

# Build datasets
df_suma = pd.DataFrame(
        {
            'Name': names,
            'Value': sum_objects
        })

nums = np.arange(int(len(persons)))
df_datos = pd.DataFrame(
        {
            'Names': nums,
            'Persons': persons, 
            'Bycicles': bycicles, 
            'Cars': cars,
            'Motorbikes': motorbikes,
            'Busses': busses, 
            'Trains': trains, 
            'Trucks': trucks, 
            'Trafics': trafics,
            'Fires': fires, 
            'Stops': stops,
            'Parks': parks,
        }
)

# ------------------ STATISTICS --------------------
for i in range(len(sum_objects)):
    print("----", names[i], "----")

    mean_objects = statistics.mean(object_count[:, i])
    mean_all.append(mean_objects)
    print("Mean: ", mean_objects)

    median_objects = statistics.median(object_count[:, i])
    print("Median: ", median_objects)

    mode_objects = statistics.mode(object_count[:, i])
    print("Mode: ", mode_objects)

    stdev_objects = statistics.stdev(object_count[:, i])
    print("StDev: ", stdev_objects)

    var_objects = statistics.variance(object_count[:, i])
    print("Variance: ", var_objects)

# ------------------ TOTAL COUNT --------------------
print('\n')
print("----Total count:----")
print(names)
print(sum_objects)

# ------------------ GRAPHS --------------------
# ------- Bar Plot -------
x = np.arange(len(sum_objects))
colors = cm.rainbow(np.linspace(0, 1, 20))

plt.bar(x, sum_objects, color= colors)
plt.xticks(x, names, size=16, rotation=90)
plt.yticks(size=18)
plt.title('Number of objects detected', size=20)
plt.ylabel('Amount', size=20)
plt.xlabel('Objects', size=20)
plt.show()

# # ------ All Bar Plots ------
# fig, axs = plt.subplots(2, 6)

# count = 0
# for i in range(2):
#     for j in range(6):
#         if count < 11:
#             axs[i, j].bar(nums, object_count[:, count])
#             axs[i, j].set_title(names[count])
#             axs[i, j].axhline(y=mean_all[count], color="grey", linestyle ="--")
#         else:
#             axs[i, j].bar(x, sum_objects)
#             axs[i, j].set_title('Sum')
#         count += 1
# plt.show()

# # ------ Spider Plot ------
# nums = [*nums, nums[0]]

# persons = [*persons, persons[0]]
# bycicles = [*bycicles, bycicles[0]]
# cars = [*cars, cars[0]]
# motorbikes = [*motorbikes, motorbikes[0]]
# busses = [*busses, busses[0]]
# trains = [*trains, trains[0]]
# trucks = [*trucks, trucks[0]]
# trafics = [*trafics, trafics[0]]
# fires = [*fires, fires[0]]
# stops = [*stops, stops[0]]
# parks = [*parks, parks[0]]

# label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(persons))

# plt.figure(figsize=(6, 6))
# plt.subplot(polar=True)

# plt.plot(label_loc, persons, label='Persons')
# plt.plot(label_loc, bycicles, label='Bycicles')
# plt.plot(label_loc, cars, label='Cars')
# plt.plot(label_loc, motorbikes, label='Motorbikes')
# plt.plot(label_loc, busses, label='Busses')
# plt.plot(label_loc, trains, label='Trains')
# plt.plot(label_loc, trucks, label='Trucks')
# plt.plot(label_loc, trafics, label='Trafic lights')
# plt.plot(label_loc, fires, label='Fire hydrants')
# plt.plot(label_loc, stops, label='Stop signs')
# plt.plot(label_loc, parks, label='park meters')

# plt.title('Object comparison', size=20, y=1.05)
# plt.thetagrids(np.degrees(label_loc), labels=nums)
# plt.legend()
# plt.show()

# # # ------ Circular Bar Plot ---------
# # Initialize the figure
# plt.figure(figsize=(6,6))
# ax = plt.subplot(111, polar=True)

# # Constants = parameters controling the plot layout:
# lowerLimit = 10
# labelPadding = 4

# # Compute max and min in the dataset
# max = df_suma['Value'].max()

# slope = (max - lowerLimit) / max
# heights = slope * df_suma.Value + lowerLimit

# # Compute the width of each bar. In total we have 2*Pi = 360°
# width = 2*np.pi / len(df_suma.index)

# # Compute the angle each bar is centered on:
# indexes = list(range(1, len(df_suma.index)+1))
# angles = [element * width for element in indexes]

# # Draw bars
# bars = ax.bar(
#     x=angles, 
#     height=heights, 
#     width=width, 
#     bottom=lowerLimit,
#     linewidth=2, 
#     edgecolor="white",
#     color="#61a4b2",
# )

# # Set the labels
# ax.set_xticks(angles)
# ax.set_xticklabels(names, size=18)

# # Add labels
# for bar, angle, height, label in zip(bars,angles, heights, df_suma["Name"]):

#     # Labels are rotated. Rotation must be specified in degrees
#     rotation = np.rad2deg(angle)

#     # Flip some labels upside down
#     alignment = ""
#     if angle >= np.pi/2 and angle < 3*np.pi/2:
#         alignment = "right"
#         rotation = rotation + 180
#     else: 
#         alignment = "left"
# plt.title('Number of objects detected', size=20, y=1.05)
# plt.show()

# # -------- Box plot ---------
# sns.boxplot(data=df_datos.loc[:, ['Persons', 'Bycicles', 'Cars', 'Motorbikes', 'Busses', 'Trains', 'Trucks', 'Trafics', 'Fires', 'Stops', 'Parks']])
# plt.show()

# ------- Heat map --------
sns.heatmap(df_datos.loc[:, ['Persons', 'Bycicles', 'Cars', 'Motorbikes', 'Busses', 'Trains', 'Trucks', 'Trafics', 'Fires', 'Stops', 'Parks']])
plt.yticks(frames, time)
plt.show()

# ------- Line plot --------
plt.plot(df_datos.loc[:, ["Names"]], df_datos.loc[:, ['Persons', 'Bycicles', 'Cars', 'Motorbikes', 'Busses', 'Trains', 'Trucks', 'Trafics', 'Fires', 'Stops', 'Parks']], label=names)
plt.xticks(frames, time)
plt.legend(), plt.xlabel("Time (sec)"), plt.ylabel("Count")
plt.show()

