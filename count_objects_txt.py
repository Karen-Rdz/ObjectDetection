import numpy as np

f = open("result.txt", "r")

car_count = person_count = trafic_count = image_count = bus_count = truck_count = stop_count = motor_count = 0

for x in f:
    ini = x[0:3]
    if ini == "dat":
        if image_count == 0:
            pass
        elif image_count == 1:
            object_count = np.array([[car_count, bus_count, truck_count, motor_count, person_count, trafic_count, stop_count]])
        else:
            result = np.array([[car_count, bus_count, truck_count, motor_count, person_count, trafic_count, stop_count]])     
            object_count = np.concatenate((object_count, result), axis = 0)
        car_count = person_count = trafic_count = bus_count = truck_count = stop_count = motor_count = 0
        image_count += 1

    elif ini == "car": car_count += 1
    elif ini == "bus": bus_count += 1
    elif ini == "tru": truck_count += 1
    elif ini == "mot": motor_count += 1
    elif ini == "per": person_count += 1
    elif ini == "tra": trafic_count += 1
    elif ini == "sto": stop_count += 1
    
result = np.array([[car_count, bus_count, truck_count, motor_count, person_count, trafic_count, stop_count]])
object_count = np.concatenate((object_count, result), axis = 0)

f.close()
print("car, bus, truck, motorbike, person, traffic light, stop sign")
print(object_count)