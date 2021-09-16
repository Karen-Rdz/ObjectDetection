import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.core.fromnumeric import mean
import pandas as pd
from math import pi
import statistics
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Open json file with YOLO results
f = open('result.json')
data = json.load(f)

# Variables
car_count = person_count = trafic_count = image_count = bus_count = truck_count = stop_count = motor_count = fire_count = bycicle_count = train_count = park_count = 0
object_count = np.zeros((1, 7))

# Count objects
for image in data:
    for object in image["objects"]:

        if object["confidence"] > 0.4:
            if object["class_id"] == 0: person_count += 1
            elif object["class_id"] == 1: bycicle_count += 1
            elif object["class_id"] == 2: car_count += 1
            elif object["class_id"] == 3: motor_count += 1
            elif object["class_id"] == 5: bus_count += 1
            elif object["class_id"] == 6: train_count += 1
            elif object["class_id"] == 7: truck_count += 1
            elif object["class_id"] == 9: trafic_count += 1
            elif object["class_id"] == 10: fire_count += 1
            elif object["class_id"] == 11: stop_count += 1
            elif object["class_id"] == 12: park_count += 1

    if image["frame_id"] == 1:
        object_count = np.array([[person_count, bycicle_count, car_count, motor_count, bus_count, train_count, truck_count, trafic_count, fire_count, stop_count, park_count]])
    else:  
        result = np.array([[person_count, bycicle_count, car_count, motor_count, bus_count, train_count, truck_count, trafic_count, fire_count, stop_count, park_count]])     
        object_count = np.concatenate((object_count, result), axis = 0)

    car_count = person_count = trafic_count = bus_count = truck_count = stop_count = motor_count = fire_count = bycicle_count = train_count = park_count = 0

# Finish reading json file
f.close()
print("person, bycicle, car, motorbike, bus, train, truck, trafic light, fire hydrant, stop sign, parking meter")
print(object_count)

# Variables
suma = np.sum(object_count, axis=0)
names = ['person', 'bycicle', 'car', 'motorbike', 'bus', 'train', 'truck', 'trafic light', 'fire hydrant', 'stop sign', 'parking meter']
nums = np.arange(len(object_count[:, 0]))
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
            'Value': suma
        })

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

df_coordinates = pd.read_json('registers.json')

# ------------------ STATISTICS --------------------
for i in range(len(suma)):
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

# ------------------ GRAPHS --------------------
# ------- Bar Plot -------
x = np.arange(len(suma))
colors = cm.rainbow(np.linspace(0, 1, 20))

plt.bar(x, suma, color= colors)
plt.xticks(x, names)
plt.show()

# ------ All Bar Plots ------
fig, axs = plt.subplots(2, 6)

count = 0
for i in range(2):
    for j in range(6):
        if count < 11:
            axs[i, j].bar(nums, object_count[:, count])
            axs[i, j].set_title(names[count])
            axs[i, j].axhline(y=mean_all[count], color="grey", linestyle ="--")
        else:
            axs[i, j].bar(x, suma)
            axs[i, j].set_title('Sum')
        count += 1
plt.show()

# ------ Spider Plot ------
nums = [*nums, nums[0]]

persons = [*persons, persons[0]]
bycicles = [*bycicles, bycicles[0]]
cars = [*cars, cars[0]]
motorbikes = [*motorbikes, motorbikes[0]]
busses = [*busses, busses[0]]
trains = [*trains, trains[0]]
trucks = [*trucks, trucks[0]]
trafics = [*trafics, trafics[0]]
fires = [*fires, fires[0]]
stops = [*stops, stops[0]]
parks = [*parks, parks[0]]

label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(persons))

plt.figure(figsize=(6, 6))
plt.subplot(polar=True)

plt.plot(label_loc, persons, label='Persons')
plt.plot(label_loc, bycicles, label='Bycicles')
plt.plot(label_loc, cars, label='Cars')
plt.plot(label_loc, motorbikes, label='Motorbikes')
plt.plot(label_loc, busses, label='Busses')
plt.plot(label_loc, trains, label='Trains')
plt.plot(label_loc, trucks, label='Trucks')
plt.plot(label_loc, trafics, label='Trafic lights')
plt.plot(label_loc, fires, label='Fire hydrants')
plt.plot(label_loc, stops, label='Stop signs')
plt.plot(label_loc, parks, label='park meters')

plt.title('Object comparison', size=20, y=1.05)
lines, labels = plt.thetagrids(np.degrees(label_loc), labels=nums)
plt.legend()
plt.show()

# ------ Circular Bar Plot ---------
# Initialize the figure
plt.figure(figsize=(6,6))
ax = plt.subplot(111, polar=True)

# Constants = parameters controling the plot layout:
lowerLimit = 10
labelPadding = 4

# Compute max and min in the dataset
max = df_suma['Value'].max()

slope = (max - lowerLimit) / max
heights = slope * df_suma.Value + lowerLimit

# Compute the width of each bar. In total we have 2*Pi = 360°
width = 2*np.pi / len(df_suma.index)

# Compute the angle each bar is centered on:
indexes = list(range(1, len(df_suma.index)+1))
angles = [element * width for element in indexes]

# Draw bars
bars = ax.bar(
    x=angles, 
    height=heights, 
    width=width, 
    bottom=lowerLimit,
    linewidth=2, 
    edgecolor="white",
    color="#61a4b2",
)

# Set the labels
ax.set_xticks(angles)
ax.set_xticklabels(names, size=12)

# Add labels
for bar, angle, height, label in zip(bars,angles, heights, df_suma["Name"]):

    # Labels are rotated. Rotation must be specified in degrees
    rotation = np.rad2deg(angle)

    # Flip some labels upside down
    alignment = ""
    if angle >= np.pi/2 and angle < 3*np.pi/2:
        alignment = "right"
        rotation = rotation + 180
    else: 
        alignment = "left"
plt.show()

# -------- Box plot ---------
sns.boxplot(data=df_datos.loc[:, ['Persons', 'Bycicles', 'Cars', 'Motorbikes', 'Busses', 'Trains', 'Trucks', 'Trafics', 'Fires', 'Stops', 'Parks']])
plt.show()

# ------- Heat map --------
sns.heatmap(df_datos.loc[:, ['Persons', 'Bycicles', 'Cars', 'Motorbikes', 'Busses', 'Trains', 'Trucks', 'Trafics', 'Fires', 'Stops', 'Parks']])
plt.show()

# ---------- Map -----------
# Creating DataFrame
df_all = pd.DataFrame(columns=['latitude', 'longitude', 'object', 'count'])
names_all = []
for i in range(0, len(object_count[:, 0])):
    for j in range(0, len(object_count[0, :])):
        if object_count[i, j] != 0:
            names_all.append(names[j])
            df = pd.DataFrame(np.array([[df_coordinates["latitude"][i], df_coordinates["longitude"][i], j, object_count[i, j]]]), columns=['latitude', 'longitude', 'object', 'count'])
            result = pd.concat([df_all, df], ignore_index=True, sort=False)
            df_all = result
# Creating map
token = "pk.eyJ1Ijoia2FyZW5yZHoiLCJhIjoiY2tyeTQ3dWNtMHdsYzJuazNjcTFjcWl3aSJ9.ix7QosHuttK1g-md6ZabDQ"
fig = px.scatter_mapbox(df_all, lat="latitude", lon="longitude",
                        color = 'object', size = "count", zoom=16, size_max=20, hover_name=names_all)
                        # height= width of image
fig.update_layout(coloraxis_colorbar=dict(
    title="Objects",
    tickvals=nums,
    ticktext=names,
))
# fig.update_layout(mapbox_style="open-street-map")                 # Discomment for Open Street Map
# fig.update_layout(mapbox_style="dark", mapbox_accesstoken=token)  # Discomment for using MapBox dark
fig.update_layout(mapbox_accesstoken=token)    # Discomment for using MapBox 
fig.update_layout(margin={"r":15,"t":15,"l":15,"b":15})
fig.show()