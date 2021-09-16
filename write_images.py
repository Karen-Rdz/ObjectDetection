f = open("data/train.txt", "w")

NUM_IMAGES = 181

for i in range(NUM_IMAGES+1):
    f.write("data/Test17/img" + str(i) + ".jpg\n")

f.close()