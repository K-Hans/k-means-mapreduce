import sys
import os
import json
from numpy import ceil
from pyspark import SparkContext
from pyspark.sql import SparkSession
from point import Point
import time
import math

def init_centroids(dataset, k):
    start_time = time.time()
    initial_centroids = dataset.takeSample(False, k)
    print("init centroid execution:", len(initial_centroids), "in", (time.time() - start_time), "s")
    return initial_centroids

def get_distances(p): # This function calculates the distances between a given point and all the centroids
    centroids = centroids_broadcast.value
    distances_list = [0] * len(centroids)
    for i in range(len(centroids)):
       distance = p.distance(centroids[i], distance_broadcast.value)
       distances_list[i] = distance
    return (distances_list, p)

def assign(l):  # This function assigns a point to the nearest centroid without violating the maximum size criterion
    distances = l[0]
    p = l[1]
    c = min(range(len(distances)), key=distances.__getitem__) # get the index of the shortest distance
    i = 0
    while sizes[c] > max_size and i < len(sizes):
        distances[c] = float("inf")
        c = min(range(len(distances)), key=distances.__getitem__)
        i +=1
    sizes[c] +=1
    return (c, p)

def stopping_criterion(new_centroids, threshold):
    old_centroids = centroids_broadcast.value
    for i in range(len(old_centroids)):
        check = old_centroids[i].distance(new_centroids[i], distance_broadcast.value) <= threshold
        if check == False:
            return False
    return True

if __name__ == "__main__":
    start_time = time.time()
    if len(sys.argv) != 3:
        print("Number of arguments not valid!")
        sys.exit(1)

    with open(os.path.join(os.path.dirname(__file__),"./config.json")) as config:
        parameters = json.load(config)["configuration"][0]

    INPUT_PATH = str(sys.argv[1])
    OUTPUT_PATH = str(sys.argv[2])
    
    spark = SparkSession.builder.appName("PySparkkmeans").getOrCreate()
    sc = spark.sparkContext
    #sc = SparkContext("yarn", "Kmeans")
    sc.setLogLevel("ERROR")
    sc.addPyFile(os.path.join(os.path.dirname(__file__),"./point.py")) # It's necessary, otherwise the spark framework doesn't see point.py

    print("\n***START****\n")

    points = sc.textFile(INPUT_PATH).map(Point).cache()
    initial_centroids = init_centroids(points, k=parameters["k"])
    distance_broadcast = sc.broadcast(parameters["distance"])
    centroids_broadcast = sc.broadcast(initial_centroids)

    nPoints = points.count()
    k=parameters["k"]
    max_size = math.ceil(nPoints / k) # The maximum cluster size
    sizes = [0] * k

    stop, n = False, 0
    while True:
        print("--Iteration n. {itr:d}".format(itr=n+1), end="\r", flush=True)
        distances_rdd = points.map(get_distances)  # Get the distances list for each point
        l = distances_rdd.collect()
        assigned = []
        for i in range(len(l)):    # assign each point to its nearest centroid and store the assignments in assigned[]
            assigned.append(assign(l[i]))
        sizes = [0] * k
        cluster_assignment_rdd = sc.parallelize(assigned) # converting assigned[] to an RDD
        assigned.clear()
        l.clear()
        clusters_sizes = cluster_assignment_rdd.countByKey() # count the sizes of the clusters
        sum_rdd = cluster_assignment_rdd.reduceByKey(lambda x, y: x.sum(y))
        centroids_rdd = sum_rdd.mapValues(lambda x: x.get_average_point()).sortByKey(ascending=True)
        new_centroids = [item[1] for item in centroids_rdd.collect()]
        stop = stopping_criterion(new_centroids,parameters["threshold"])

        n += 1
        if(stop == False and n < parameters["maxiteration"]):
            centroids_broadcast = sc.broadcast(new_centroids)
        else:
            break
    
    with open(OUTPUT_PATH, "w") as f:
        for centroid in new_centroids:
            f.write(str(centroid) + "\n")

    execution_time = time.time() - start_time
    print("\nexecution time:", execution_time, "s")
    print("average time per iteration:", execution_time/n, "s")
    print("n_iter:", n)
    print("clusters sizes: ", clusters_sizes) # print the clusters sizes
