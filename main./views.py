"""
Visualisation of ACO applied to route optimistion
Final 3rd year individual project for ECM3401
This is a django system that uses the leaflet.js package to display map dataset
ACO implementaion is used to find optimal paths between valid targets in
an area
"""
import random
import time
from queue import PriorityQueue
from math import inf
from django.shortcuts import render
import folium
import openrouteservice as ors
from django.http import HttpResponse, JsonResponse
# graph dependency
import osmnx as ox
import networkx as nx
import numpy as np

from IPython.display import IFrame, display
#import matplotlib.pyplot as plt




# OpenRouteService api key
ors_key = '5b3ce3597851110001cf624844229b68a7af4311b243a28407590a15'


def index(request):
    """
    Index page render
    """
    #client
    client = ors.Client(key=ors_key)
    #
    #map
    m = folium.Map(location=[50.73207170509859,-3.5170272043617445,],
                    tiles='cartodbpositron',
                    zoom_start=5,
                    max_bounds=True)

    folium.ClickForMarker().add_to(m)

    m = m._repr_html_()

    context={
        'display_map' : m
    }

    return render(request, 'index.html', context)



def visualisation(request):
    """
    Visualisation page render
    """
    print(PATH.origin, PATH.dest)
    context={

            'bbox' : PATH.visualbbox,
            'origin': PATH.origin,
            'dest': PATH.dest
        }

    return render(request, 'visual.html', context)

def reset_path_run(request):
    """
    Reset path get request
    """
    PATH.stop()
    return HttpResponse('Success')

def vis_logic(request):
    """
    Parameter get request for search algorthm
    """
    type_of_al = request.GET.getlist("a_type")
    evaporation_val = request.GET.get("evap_val")
    ant_num = request.GET.get("ant_num")
    evaluations = request.GET.get("eval_num")
    path_toggle = request.GET.get("path_toggle")
    max_p = request.GET.get("max_p")
    min_p = request.GET.get("min_p")
    beta = request.GET.get("beta_val")

    #print(max_p, min_p, beta)

    print(path_toggle)
    if path_toggle == 'false':
        PATH.toggle_pathNum = False
    if path_toggle == 'true':
        PATH.toggle_pathNum = True
    print("Starting visual")

    PATH.stop()

    #initialise the envrionement class that encompasses all the logic for pathfinding
    E = Environment(PATH.logicbbox, PATH.origin, PATH.dest, int(ant_num),
    float(evaporation_val), int(evaluations), PATH.area_type,
    float(beta), float(max_p), float(min_p))

    if E.valid_data[0] == False:
        PATH.area_error = True
        PATH.error_message = E.valid_data[1]
        print("Area error, osmnx data not retreivable for area")
        return HttpResponse("Error")
    else:
        print("done processing")

        if type_of_al == ['Dijkstra']:
            dijkstra_control(E)
        elif type_of_al == ['Depth-first-search']:
            d_first_search_control(E)
        else:
            aco_control(E, type_of_al)

        return HttpResponse("Success")




def vis_finish(request):
    """
    Visualisation is finished
    """
    PATH.running = False
    return HttpResponse('Success')


def points_api(request):
    """
    Data get request: variables are updated and the front-end takes the data to
    render on the leaflet map
    """
    return JsonResponse({'array': PATH.path,
    'running' : PATH.running,
    'runType': PATH.run_type,
    'path_num' : PATH.toggle_pathNum,
    'fitness' : PATH.fitness_ofpath,
    'done': PATH.done,
    'Load': PATH.area_error,
    'error_message': PATH.error_message})


def grouped(iterable, n):

    """
    group lat lang data array create
    """
    a = zip(*[iter(iterable)]*n)
    return a


def simplfy(array):
    """
    simplfy the data recievd from the index page
    """
    array = grouped(array, 2)
    sort_array = []
    for x, y in array:
        temp = []
        temp.append(float(x))
        temp.append(float(y))
        sort_array.append(temp)

    return sort_array


def locations(request):
    """
    Locations view that recieves the area bounds data and target point data
    """
    #get data
    points_array = request.GET.getlist("locationArray[]")
    bbox_array = request.GET.getlist("bboxArray[]")
    area_type = request.GET.get("areaType")

    PATH.reset()

    points_array = simplfy(points_array)
    bbox_array = simplfy(bbox_array)
    PATH.origin = points_array[0]
    PATH.dest = points_array[1]

    PATH.area_type = area_type.lower()

    #print(PATH.area_type)
    bbox_array.append(points_array[0])
    bbox_array.append(points_array[1])
    #print(points_array, bbox_array)

    logic(bbox_array)
    PATH.origin.reverse()
    PATH.dest.reverse()

    return HttpResponse('Success')


def logic(main_loc_array):
    """
    Creation of points in order to determin the area bounds for data available
    """
    point_object_array = []
    for i in main_loc_array:
        i.reverse()
        p = Pointobj(i)
        point_object_array.append(p)

    E = EnvCreate(PATH.area_type)

    bounding_iArea = E.get_isochrones(point_object_array)

    PATH.logicbbox = bounding_iArea

    print("here", bounding_iArea)

    PATH.visualbbox = [[bounding_iArea[0][1], bounding_iArea[0][0]],
    [bounding_iArea[1][1], bounding_iArea[1][0]]]

"""
________________________________________________________________________________
Algorithm control - these fucntion dictate the runtime function for
path display and visualisation runtime. Data is output from the search
algorithm, and the points_api() request data is updated. The next call request
will take this new data to render. Once rendered, the next iteration is run

"""


def dijkstra_control(E):
    """
    Dijkstra control
    """
    best_path, total_vis, fitness = E.dijkstra_run()

    if PATH.running == False:
        PATH.fitness_ofpath = fitness
        PATH.run_type = True
        PATH.path = total_vis
        PATH.running = True

    time.sleep(1)
    PATH.run_type = False
    PATH.path = best_path
    #PATH.running = True
    time.sleep(1)
    PATH.done = True


def d_first_search_control(E):
    """
    Depth first search control
    """
    found_ant, fitness = E.initial_search()
    # for i in found_ant.visited_nodes:
    #     print(i.node)
    #     print(i.adj_list)

    if PATH.running == False:
        PATH.fitness_ofpath = fitness
        PATH.run_type = True
        PATH.path = found_ant.path
        PATH.running = True

    time.sleep(1)
    PATH.run_type = False
    PATH.path = found_ant.xy_data
    #PATH.running = True
    time.sleep(1)
    PATH.done = True


def aco_control(E, type_of_al):
    """ACO control for each algorithm
    Checks which ACO algorithm to run.
    """
    #sub_toggle is to determin whether multiple paths are to be shown or just one
    sub_toggle = PATH.toggle_pathNum
    if type_of_al == ['ACO-Basic']:
        if sub_toggle:
            basic = True
            i = 0
            while i < E.fitness_evaluations:
                if PATH.running == False:
                    return_paths, best_fitness = E.ant_optimisation_lead(basic)
                    PATH.fitness_ofpath = best_fitness
                    if len(return_paths) > 1:
                        if sub_toggle:
                            PATH.toggle_pathNum = True
                            PATH.path = return_paths
                        else:
                            PATH.path = return_paths[0]
                        PATH.running = True
                        i+=1
                    else:
                        PATH.toggle_pathNum = False
                        PATH.path = return_paths[0]
                        PATH.running = True
                        i+=1
                else:
                    time.sleep(0.05)

            PATH.toggle_pathNum = False
            PATH.path = return_paths[0]
            #PATH.running = True
            time.sleep(1)
            PATH.done = True

        else:
            i = 0
            #PATH.running = True
            while i < E.fitness_evaluations:
                if PATH.running == False:
                    return_path, best_fitness = E.ant_optimisation()
                    PATH.path = return_path
                    PATH.fitness_ofpath = best_fitness
                    PATH.running = True
                    i+=1
                else:
                    time.sleep(0.05)
            PATH.path = return_path
            #PATH.running = True
            time.sleep(1)
            PATH.done = True


    elif type_of_al == ['ACO-Leaders']:
        basic = False
        i = 0
        #PATH.running = True
        while i < E.fitness_evaluations:
            if PATH.running == False:
                return_paths, best_fitness = E.ant_optimisation_lead(basic)
                PATH.fitness_ofpath = best_fitness
                if len(return_paths) > 1:
                    if sub_toggle:
                        PATH.toggle_pathNum = True
                        PATH.path = return_paths
                    else:
                        PATH.path = return_paths[0]
                    PATH.running = True
                    i+=1
                else:
                    PATH.toggle_pathNum = False
                    PATH.path = return_paths[0]
                    PATH.running = True
                    i+=1
            else:
                time.sleep(0.05)


        PATH.toggle_pathNum = False
        PATH.path = return_paths[0]
        #PATH.running = True
        time.sleep(1)
        PATH.done = True
    elif type_of_al == ['ACO-Astar-Local']:
        i = 0
        #PATH.running = True
        while i < E.fitness_evaluations:
            if PATH.running == False:
                return_paths, best_fitness = E.ant_optimisation_astar()
                PATH.fitness_ofpath = best_fitness
                if len(return_paths) > 1:
                    if sub_toggle:
                        PATH.toggle_pathNum = True
                        PATH.path = return_paths
                    else:
                        PATH.path = return_paths[0]
                    PATH.running = True
                    i+=1
                else:
                    PATH.toggle_pathNum = False
                    PATH.path = return_paths[0]
                    PATH.running = True
                    i+=1
            else:
                time.sleep(0.05)


        PATH.toggle_pathNum = False
        PATH.path = return_paths[0]
        #PATH.running = True
        time.sleep(1)
        PATH.done = True

    #print("#########", PATH.path)


class PATH():
    """
    Path class

    Contains global data variables that allows the edit of data to be parsed to the
    front-end render. Each variable is updated respectively, where the data in
    points_api() context data is set. The next ajax call request will recieve this
    new data.
    """
    path = []
    visualbbox = []
    logicbbox = []
    origin = []
    dest = []
    running = False
    done = False
    run_type = False
    toggle_pathNum = False
    area_type = 'drive'
    fitness_ofpath = 0
    area_error = False
    error_message = " "

    """
    Reset the variables
    """
    def reset():
        PATH.path = []
        PATH.visualbbox = []
        PATH.logicbbox = []
        PATH.origin = []
        PATH.dest = []
        PATH.running = False
        PATH.done = False
        PATH.run_type = False
        PATH.toggle_pathNum = False
        PATH.fitness_ofpath = 0
        PATH.area_error = False
        PATH.error_message = " "

    """
    Stop the current run of the
    """
    def stop():
        PATH.path = []
        PATH.running = False
        PATH.done = False
        PATH.run_type = False



class Pointobj():
    """
    Class that creates points objects for the cration of the bound areas
    """
    def __init__(self, point):
        self.point = point
        self.isochrone_bounds = []
        self.possible_locations = []
        self.distance_m = []


class EnvCreate():
    """
    Class that finds the specifc bounds area that has data.
    A list of points is recieved, and isochrone areas are found for each point
    Then the total actual area boundsa area found
    """
    def __init__(self, area_type):
        """
        Creation of variables for the area
        """
        self.client = ors.Client(key=ors_key)
        self.points_array = []
        self.boundingBox = []
        self.minx, self.miny = float("inf"), float("inf")
        self.maxx, self.maxy = float("-inf"), float("-inf")
        if area_type == 'drive':
            self.areaType = 'driving-car'
        elif area_type == 'walk':
            self.areaType = 'foot-walking'
        else:
            self.areaType = 'cycling-regular'


    def get_isochrones(self, loc_array):
        """All points are used to generate isochrone areas"""
        tot_arr = []
        for point in loc_array:
            isochrone = self.client.isochrones(locations=[point.point], range_type='distance', range=[500], attributes=['area'], profile=self.areaType)
            point.isochrone_bounds = isochrone['features'][0]['geometry']['coordinates'][0]

        self.bounds(loc_array)

        return self.boundingBox


    def bounds(self, points):
        """Finds the max-min of the bounds area"""
        if len(points) == 0:
            raise ValueError("Can't compute bounding box of empty list")

        self.minx, self.miny = float("inf"), float("inf")
        self.maxx, self.maxy = float("-inf"), float("-inf")

        for node in points:
            listNode = node.isochrone_bounds
            for item in listNode:
                x = item[0]
                y = item[1]
                # Set min coords
                if x < self.minx:
                    self.minx = x
                if y < self.miny:
                    self.miny = y
                # Set max coords
                if x > self.maxx:
                    self.maxx = x
                if y > self.maxy:
                    self.maxy = y

        self.boundingBox.append([self.minx, self.miny])
        self.boundingBox.append([self.maxx, self.maxy])
        print(self.boundingBox)

    def width(self):
        """width"""
        return self.maxx - self.minx

    def height(self):
        """height"""
        return self.maxy - self.miny


class ANT():
    """Ant object class, with arrays to hold the path data that the ant traversed"""
    def __init__(self, ant_number):
        self.n = ant_number
        self.visited_nodes = []
        self.path = []
        self.path_index = []
        self.xy_data = []
        self.path_fitness = 0
        self.starting_node = None
        self.target = False
        self.path_length_array = []

    def reset(self):
        """resets the ant arrays"""
        self.visited_nodes = []
        self.path = []
        self.path_index = []
        self.path_fitness = 0
        self.target = False
        self.xy_data = []
        self.path_length_array = []


class Node():
    """Node object for the graph Environment
    Adjacncy list contian the node number
    Location contians the index of where the adj node exists in the graph
    Pheromone contians the pheromone to traverse to that node
    """
    def __init__(self, node_number):
        self.node = node_number
        self.adj_list = []
        self.loc = []
        self.p = []
        self.seen = False


class Environment():
    """Environment class contians all the logic and envrionement arrays for the system

    Initialised with the area box, the orgin and destination points, number of ants,
    evapouration value, number of fitness evaluations, the type of area, beta rate and
    max-min vlaues for the pheromone
    """
    def __init__(self, bounding_box, origin_loc, destination_loc, ant_num, e_inp, fit_elvaluations, area_type, beta, max_p, min_p):
        self.valid_data = [True, None]
        #self.vis_running = False
        self.fitness_evaluations = fit_elvaluations
        self.numeber_ofANTobjects = ant_num
        self.total_graph_length = 0
        self.bbox = bounding_box
        self.graph = None
        #self.gdf_edges = None
        self.nodeObjectList = []
        self.antObjectList = []
        #//
        self.node_list = []
        self.adj_matrix = [[]]
        self.e_val = e_inp
        self.beta = beta
        self.max_p = max_p
        self.min_p = min_p
        self.area_type = area_type
        #//
        self.target_node = destination_loc
        self.start_node = origin_loc
        self.found = False
        #//
        self.best_path = [[]]
        self.seen_nodes = []
        #//check for Environment load errors
        response, error_txt = self.area()
        if response == 200:
            self.ant_innit()
            self.map_fitness()
        else:
            self.valid_data = [False, error_txt]
        #self.main_run()


    def area(self):
        """Area initialise

        The bounding box area is set and osmnx data is requested and used to create
        the graph
        """
        if len(self.bbox) == 0:
            print("Failed to load, area error")
            return 500, "Bounding box error"
        #self.bbox = [[-1.78, 49.93], [-1.80, 49.92]] this is a bad area test
        north, south, east, west = self.bbox[1][1], self.bbox[0][1], self.bbox[0][0],  self.bbox[1][0]
        try:
            #error handel here for the osmnx data
            self.graph = ox.graph_from_bbox(north, south, east, west, network_type=self.area_type)
            self.node_list = list(self.graph)
            self.adj_matrix = nx.adjacency_matrix(self.graph, nodelist=None, dtype=None, weight='weight')
            self.adj_matrix = self.adj_matrix.toarray(order=None, out=None)
        except:
            return 500, "OSMnx data error"

        self.start_node = ox.distance.nearest_nodes(self.graph, self.start_node[1], self.start_node[0])
        self.target_node = ox.distance.nearest_nodes(self.graph, self.target_node[1], self.target_node[0])

        try:
            #check a route is possible
            route = nx.shortest_path(self.graph, self.start_node, self.target_node, weight='length')
        except:
            print("Route not possible between points")
            return 500, "Route error"

        print("Len of nodes: ", len(self.graph.nodes))
        self.gen_nodes()
        return 200, " "


    def set_targets(self):
        """Set target nodes for the ACO ants"""
        c = 0
        start_li = []
        for node in self.nodeObjectList:
            if self.start_node == node.node:
                start_li = [node,c]
            if self.target_node == node.node:
                self.target_li = node
            c+=1

        return start_li

    def ant_innit(self):
        """Init ant creation"""
        s_node = self.set_targets()
        for i in range(self.numeber_ofANTobjects):
            ant = ANT(i)
            ant.starting_node = s_node
            self.antObjectList.append(ant)

#         for i in self.nodeObjectList:
#             print(i.node, i.adj_list, i.loc, i.p)

    def map_fitness(self):
        """Total map fitness, all egdes are added"""
        for node in self.nodeObjectList:
            node_dict = self.graph.__getitem__(node.node)
            for adj in node.adj_list:
                self.total_graph_length += node_dict[adj][0]['length']

        print("Total fit:", self.total_graph_length)

    def links(self, array):
        """Takes array where index which are not 0 show which node are adjacent
        Ex =  [[0,0,0,1,0]], ....
        node0, the first in the array, has adjacency [0,0,0,1,0]
        therefore node0 is adjacent to node3
        returns the index
        """
        location = np.where(array != 0)
        for list_l in location:
            return list_l

    def neighbours(self, node_number):
        """gets a node number to get which other nodes by number its adjacent to """
        adjacent_nodes = self.graph.__getitem__(node_number)
        pos_arr = list(adjacent_nodes.keys())
        return np.sort(pos_arr)

    def pheromone(self, adj_array):
        """Init of pheromone values, randomised for the intial envrionement"""
        p_array = []
        for edge in adj_array:
            p_array.append(random.uniform(0.4,0.7))
        return p_array

    def gen_nodes(self):
        """Generates the graph envrionement"""
        self.nodeObjectList = []
        c = 0
        for n in self.node_list:
            #node object
            node = Node(n)
            arr = self.adj_matrix[c]
            #get the adjacency
            loc = self.links(arr)
            #index of the nodes in the list
            node.loc = loc
            adj_n = self.neighbours(n)
            #node objects adjacent to the node
            node.adj_list = adj_n
            #create pheromone values for the edges
            pher = self.pheromone(adj_n)
            node.p = pher
            #add to the list
            self.nodeObjectList.append(node)
            c+=1



    def initial_search(self):
        """DFS
        Calls fucntions to perform a depth first search. If no path is found then
        no path is returned
        """
        #gdf_nodes, gdf_edges = ox.graph_to_gdfs(self.graph,nodes=True, edges=True,node_geometry=True,fill_edge_geometry=True)
        for ant in self.antObjectList:
            self.find_init_path(ant)
            if self.found:
                print("FOUND")
                self.set_xy_data(ant)
                total_vis = []
                for node_visted in ant.path:
                    node_xy = self.graph.nodes[node_visted.node]
                    temp = []
                    temp.append(node_xy["y"])
                    temp.append(node_xy["x"])
                    total_vis.append(temp)
                ant.path = total_vis
                return ant, ant.path_fitness

        if self.found == False:
            print("Could not find a path")
            return self.antObjectList[0], self.antObjectList[0].path

    def dijkstra_run(self):
        """Djikstra

        Calls fucntions to perfrom dijkstra search
        """
        _ ,  gdf_edges = ox.graph_to_gdfs(self.graph)

        length_graph = gdf_edges['length']

        shortest_path, distance, total_order = self.dijkstra(self.graph, length_graph, self.start_node, self.target_node)

        #print("LAST", distance)

        total_vis = []
        for node_visted in total_order:
            node_xy = self.graph.nodes[node_visted]
            temp = []
            temp.append(node_xy["y"])
            temp.append(node_xy["x"])
            total_vis.append(temp)

        shortest_path, dist = self.dijkstra_xy(shortest_path)

        if distance == 0:
            distance = dist

        return shortest_path, total_vis, distance



    def show_graph(self, path):
        """Road network on matplotlib"""
        # fig, ax = ox.plot_graph_route(self.graph, path, route_linewidth=6, bgcolor='k', show=False)
        # fig.canvas.draw()
        # plt.show(block=True)

        #====================================================================================
        # G = ox.projection.project_graph(self.graph, to_crs=3857)
        # _, gdf_edges = ox.graph_to_gdfs(G)
        #
        # #print(gdf_edges)
        #
        # fig = plt.figure(figsize=(10,10))
        # ax = plt.axes()
        #
        # ax.set(facecolor = "black")
        # ax.set_title("Length of roads", fontsize=20)
        # gdf_edges[gdf_edges.length > 10].plot(ax=ax, cmap="viridis", column="length", legend=True)
        # plt.show()

    def ant_optimisation_lead(self, basic):
        """ACO Leaders
        Performs ACO leaders search
        """
        for ant in self.antObjectList:
            self.generate_ant_path(ant)
        fit_ant_list = self.best_fitness_lead()
        best_fitness = fit_ant_list[0].path_fitness
        if basic:
            self.graph_update(fit_ant_list[0])
        else:
            self.graph_update_lead(fit_ant_list)
        self.evaporate_pheromone()

        best_path_list = []

        for ant in fit_ant_list:
            best_path_list.append(self.gen_ant_xy(ant))

        self.reset()
        return best_path_list, best_fitness


    def ant_optimisation_astar(self):
        """ACO A-Star
        Performs ACO A* search
        """
        for ant in self.antObjectList:
            self.generate_antStar_path(ant)
        fit_ant_list = self.best_fitness_lead()
        best_fitness = fit_ant_list[0].path_fitness

        self.graph_update(fit_ant_list[0])
        self.evaporate_pheromone()

        best_path_list = []

        for ant in fit_ant_list:
            best_path_list.append(self.gen_ant_xy(ant))

        self.reset()

        return best_path_list, best_fitness

    def ant_optimisation(self):
        """ACO basic
        Performs the basic ACO algorthm search
        """
        for ant in self.antObjectList:
            self.generate_ant_path(ant)
        fit_ant = self.best_fitness()
        best_fitness = fit_ant.path_fitness
        self.graph_update(fit_ant)
        self.evaporate_pheromone()
        best_path = self.gen_ant_xy(fit_ant)
        # best_path = [n.node for n in fit_ant.path]
        self.reset()
        return best_path, best_fitness

    def reset(self):
        """
        Resets the ant path data for next iteration
        """
        for ant in self.antObjectList:
            ant.visited_nodes = []
            ant.path = []
            ant.path_index = []
            ant.path_fitness = 0
            ant.target = False
            ant.xy_data = []

    def target_Pupdate(self):
        """
        Updates the target to be more entising, ants should travel towards it more
        """
        for i in self.target_li.loc:
            node = self.nodeObjectList[i]
            c = 0
            for j in node.adj_list:
                if j == self.target_li.node:
                    node.p[c] = 1
                c+=1

    def evaporate_pheromone(self):
        """Evapouration of the pheromone across the whole graph
        Ensure that it doesnt fall below the minimum
        """
        for node in self.nodeObjectList:
            temp_p = []
            for pher in node.p:
                if pher*self.e_val < self.min_p:
                    temp_p.append(self.min_p)
                else:
                    temp_p.append(pher*self.e_val)
            node.p = temp_p
        self.target_Pupdate()

    def graph_update_lead(self, ant_list):
        """Updates the graph according to the ant path taken, for multiple ants"""
        for ant_obj in ant_list:
            if ant_obj.target:
                node_iter = ant_obj.starting_node[1]
                init_n = self.nodeObjectList[node_iter]
                i = 0
                for node in ant_obj.path:
                    alpha = float(1/ant_obj.path_length_array[i])
                    fitness_update = alpha + self.beta
                    choice = ant_obj.path_index[i]
                    adj_c = 0
                    for adj in node.loc:
                        if adj == choice:
                            if node.p[adj_c] + fitness_update > self.max_p:
                                node.p[adj_c] = self.max_p
                            else:
                                node.p[adj_c] += fitness_update
                        adj_c +=1
                    i+=1

    def graph_update(self, ant_obj):
        """Updates the graph according to the ant path taken"""
        fitness_update = 0
        # fitness_update = 100/ant_obj.path_fitness
        if ant_obj.target:
            node_iter = ant_obj.starting_node[1]
            init_n = self.nodeObjectList[node_iter]
            i = 0
            for node in ant_obj.path:
                alpha = float(1/ant_obj.path_length_array[i])
                fitness_update = alpha + self.beta
                choice = ant_obj.path_index[i]
                adj_c = 0
                for adj in node.loc:
                    if adj == choice:
                        if node.p[adj_c] + fitness_update > self.max_p:
                            node.p[adj_c] = self.max_p
                        else:
                            node.p[adj_c] += fitness_update
                    adj_c +=1
                i+=1

    def dijkstra_xy(self, path):
        """Dijkstra path xy coords calculation"""
        lineObject_xy = []
        total_cost = 0
        for i in range(len(path)-1):
            current_node_dict = self.node_dict(path[i])
            try:
                coords, edge_cost = self.get_lineObject(current_node_dict[path[i+1]], path[i+1])
                for j in coords:
                    lineObject_xy.append(j)
                total_cost += edge_cost
            except:
                continue

        return lineObject_xy, total_cost

    def node_dict(self, node):
        """Node dict, contianing the line data for the edges, the length
        and type of road
        """
        return self.graph.__getitem__(node)

    def gen_ant_xy(self, ant):
        """ACO path xy coords calculation"""
        for i in range(len(ant.path)-1):
            current_node_dict = self.node_dict(ant.path[i].node)
            coords, _ = self.get_lineObject(current_node_dict[ant.path[i+1].node], ant.path[i+1].node)
            for j in coords:
                ant.xy_data.append(j)

        return ant.xy_data


    def set_xy_data(self, ant):
        """XY data for DFS, as this doesn take a path route. Instead vists the nodes
        Preprocess of duplicate nodes removal. Uses an ant object to display the path
        """
        fit = 0
        list_nnn = []
        for cc in range(len(ant.visited_nodes)-1):
            if ant.visited_nodes[cc].node != ant.visited_nodes[cc+1].node:
                list_nnn.append(ant.visited_nodes[cc])
                cc+=1
            else:
                cc+=1

        list_nnn.append(ant.visited_nodes[cc])
        ant.xy_data = []
        for i in range(len(list_nnn)-1):
            current_node_dict = self.graph.__getitem__(list_nnn[i].node)
            coords, edge_cost = self.get_lineObject(current_node_dict[list_nnn[i+1].node], list_nnn[i+1].node)
            ant.path_fitness += edge_cost
            #fit += current_node_dict[list_nnn[i+1].node][0]['length']
            for j in coords:
                ant.xy_data.append(j)

    def get_lineObject(self, node_dict, node):
        """Line object data for the edge"""
        coords = []
        try:
            lineObject = node_dict[0]['geometry']

            x,y = lineObject.coords.xy
            for i in range(len(x)):
                temp = []
                temp.append(y[i])
                temp.append(x[i])
                coords.append(temp)
        except:
            node_xy = self.graph.nodes[node]
            temp = []
            temp.append(node_xy["y"])
            temp.append(node_xy["x"])
            coords.append(temp)

        length = node_dict[0]['length']
        return coords, length


    def best_fitness_lead(self):
        """Calculate the best fitness route for leaders ACO"""
        cost = self.total_graph_length
        #best_ant = None
        best_ant = self.antObjectList[0]
        #cost = self.total_graph_length

        target_list = []
        for ant in self.antObjectList:
            if ant.path_fitness == 0:
                continue
            else:
                #if ants have reached the target
                if ant.target:
                    target_list.append(ant)
                else:
                    if ant.path_fitness < cost:
                        cost = ant.path_fitness
                        best_ant = ant

        """To separate the ants which have reached the target to those which have not

        The list of ants which have are then compared.

        Only allow the top 10% of ant paths to perfrom pheromone updates
        """

        #target_list full of ants with a paths
        top_leaders_num = int(self.numeber_ofANTobjects*0.1)
        top_l = []
        if len(target_list) != 0:
            if len(target_list) > top_leaders_num:
                while len(top_l) != top_leaders_num:
                    cost = target_list[0].path_fitness
                    best_ant = target_list[0]
                    for ant in target_list:
                        if ant.path_fitness < cost:
                            cost = ant.path_fitness
                            best_ant = ant
                    target_list.remove(best_ant)
                    top_l.append(best_ant)
                return top_l
            else:
                l = len(target_list)
                while len(top_l) != l:
                    cost = target_list[0].path_fitness
                    best_ant = target_list[0]
                    for ant in target_list:
                        if ant.path_fitness < cost:
                            cost = ant.path_fitness
                            best_ant = ant
                    target_list.remove(best_ant)
                    top_l.append(best_ant)
                return top_l

        return [best_ant]

    def best_fitness(self):
        """Calculate the best fitness route for elitist ACO"""
        cost = self.total_graph_length
        #print(self.antObjectList)
        best_ant = self.antObjectList[0]

        cost = self.total_graph_length

        target_list = []
        for ant in self.antObjectList:
            if ant.path_fitness == 0:
                continue
            else:
                if ant.target:
                    target_list.append(ant)
                else:
                    if ant.path_fitness < cost:
                        cost = ant.path_fitness
                        best_ant = ant

        if len(target_list) != 0:
            cost = target_list[0].path_fitness
            best_ant = target_list[0]
            for ant in target_list:
                if ant.path_fitness < cost:
                    cost = ant.path_fitness
                    best_ant = ant

        return best_ant

    def evaluate_fitness(self, node_dict):
        """returns the length of the edge """
        edge_cost = node_dict[0]['length']
        return edge_cost

    def next_nodeObject(self, index):
        """returns the node object of the given index"""
        return self.nodeObjectList[index]

    def get_adjacentNodes(self, node):
        """returns an index choice based off the pheromone values in adjacency"""
        edges = [i for i in range(len(node.loc))]

        chosen_edge = random.choices(edges, node.p)

        return chosen_edge[0]


    def get_initadjacentNodes(self, node):
        """Gets the adjacent nodes to the current node, for DFS"""
        for j in node.loc:
            if not (self.nodeObjectList[j].seen):
                r = np.where(node.loc == j)
                return r[0][0]
        return None


    def dijkstra(self, graph, length_graph, start, end):
        """Dijkstra search algorithm"""
        def backtrace(prev, start, end, dist):
            """backtrace to the start node"""
            try:
                node = end
                path = []
                while node != start:
                    path.append(node)
                    node = prev[node]
                path.append(node)
                path.reverse()
                return path, dist[end]
            except:
                route = nx.shortest_path(self.graph, start, end, weight='length')
                #print(route)
                return route, 0

        def neighbour(curr):
            """Get adjacent nodes to current node"""
            adj_list = {}
            try:
                neigh = length_graph[curr].keys()
                for n in neigh:
                    nn = n[0]
                    adj_list[nn] = length_graph[curr][nn][0]
            except:
                return adj_list

            return adj_list

        def cost(u, v):
            """Cost of edge"""
            return length_graph[u][v][0]

        # predecessor of current node on shortest path
        prev = {}
        # initialize distances from start -> given node i.e. dist[node] = dist(start: str, node: str)
        dist = {v: inf for v in list(nx.nodes(graph))}
        # nodes we've visited
        visited = set()
        # prioritize nodes from start -> node with the shortest distance
        ## elements stored as tuples (distance, node)
        pq = PriorityQueue()

        dist[start] = 0  # dist from start -> start is zero
        pq.put((dist[start], start))

        path_list = []
        while 0 != pq.qsize():
            curr_cost, curr = pq.get()
            visited.add(curr)
            # look at curr's adjacent nodes
            path_list.append(curr)
            for neighbor in neighbour(curr):
                # if we found a shorter path
                path = dist[curr] + cost(curr, neighbor)
                if path < dist[neighbor]:
                    #print(path)
                    # update the distance
                    dist[neighbor] = path
                    # update the previous node to be prev on new shortest path
                    prev[neighbor] = curr
                    # if we haven't visited the neighbor
                    if neighbor not in visited:
                        # insert into priority queue and mark as visited
                        visited.add(neighbor)
                        pq.put((dist[neighbor],neighbor))
                    # otherwise update the entry in the priority queue
                    else:
                        # remove old
                        _ = pq.get((dist[neighbor],neighbor))
                        # insert new
                        pq.put((dist[neighbor],neighbor))
            # we are done after every possible path has been checked
        route_path, fitness_route = backtrace(prev, start, end, dist)
        return route_path, fitness_route, path_list

    def find_init_path(self, ant):
        """DFS algorithm"""
        #start at orgin
        iterator_node = ant.starting_node[0]
        ant.path.append(iterator_node)
        ant.visited_nodes.append(iterator_node)
        #iterator_node.seen = True
        for _ in range(int(len(self.graph.nodes)*2)):
            #adj nodes are every node not see, first not seen is picked
            edge_choice = self.get_initadjacentNodes(iterator_node)
            if edge_choice == None:
                #if no more edges not visted, return to the last node
                ant.visited_nodes.pop()
                if len(ant.visited_nodes) == 0:
                    break
                last = ant.visited_nodes[-1]
                iterator_node = last
            else:
                #next node to search
                next_node = self.next_nodeObject(iterator_node.loc[edge_choice])
                ant.visited_nodes.append(next_node)
                ant.path.append(next_node)
                #set to seen
                next_node.seen = True
                iterator_node = next_node
                #check its the target
                if next_node.node == self.target_node:
                    ant.visited_nodes.append(next_node)
                    self.found = True
                    break

    def generate_ant_path(self, ant):
        """ACO-Basic algorithm"""
        #starting node
        iterator_node = ant.starting_node[0]
        #energy value
        for _ in range(int(len(self.graph.nodes))):
            ant.path.append(iterator_node)
            #if ant encounters a 'dead' node, then break
            if len(iterator_node.adj_list) == 0:
                ant.reset()
                break
            #edge choice
            edge_choice = self.get_adjacentNodes(iterator_node)
            #add choice to index array, rembering which edge is taken to update the pheromone
            ant.path_index.append(iterator_node.loc[edge_choice])
            #get the next node choice
            node_choice = iterator_node.adj_list[edge_choice]
            current_node_dict = self.graph.__getitem__(iterator_node.node)
            #length of path
            edge_fitness = self.evaluate_fitness(current_node_dict[node_choice])
            #add fitness to the ant path
            ant.path_fitness += edge_fitness
            #add the fitness to determin the fitness overtime
            ant.path_length_array.append(edge_fitness)
            #get next node
            next_node = self.next_nodeObject(iterator_node.loc[edge_choice])
            #check if target point
            if next_node.node == self.target_node:
                ant.path.append(next_node)
                ant.path_index.append(iterator_node.loc[edge_choice])
                ant.path_length_array.append(edge_fitness)
                ant.target = True
                break
            #set the new node to the iterator
            iterator_node = next_node


    def euclidean_dist_vec(self, x1, x2, y1, y2):
        """euclidean distance"""
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


    def dist_fromorigin(self, node):
        """Calculates the distance from the origin"""
        curr_node = self.graph.nodes[node]
        x1 = curr_node["x"]
        y1 = curr_node["y"]

        t_node = self.graph.nodes[self.start_node]
        x2 = t_node["x"]
        y2 = t_node["y"]

        euclidean_dist = self.euclidean_dist_vec(x1, x2, y1, y2)

        return euclidean_dist


    def dist_fromtarget(self, node):
        """Calculates the distance to the target"""
        curr_node = self.graph.nodes[node]
        x1 = curr_node["x"]
        y1 = curr_node["y"]

        t_node = self.graph.nodes[self.target_node]
        x2 = t_node["x"]
        y2 = t_node["y"]

        euclidean_dist = self.euclidean_dist_vec(x1, x2, y1, y2)

        return euclidean_dist


    def a_star_funct(self, node, o_list, t_list):
        """A* fucntion to calcualte which node is better"""
        Q = 1
        fn = []
        for i in range(len(node.adj_list)):
            n_ij = Q / (o_list[i] + t_list[i])
            fn.append(n_ij)

        return fn


    def aStar_distance(self, node):
        """A* function, adds a local pheromone to determin the better path"""
        dist_o = []
        dist_t = []
        a_starli = []
        for i in node.adj_list:
            distf_origin = self.dist_fromorigin(i)
            dist_o.append(distf_origin)
            distf_target = self.dist_fromtarget(i)
            dist_t.append(distf_target)

        p_upd = self.a_star_funct(node, dist_o, dist_t)
        a_starli = [sorted(p_upd).index(x) for x in p_upd]
        a_starli = np.array(a_starli)


        pval = 0.02
        index = np.where(a_starli == 0)

        a_star_p = []
        for i in node.p:
            a_star_p.append(i)

        a_star_p[index[0][0]] += pval
        if a_star_p[index[0][0]] > self.max_p:
            a_star_p[index[0][0]] = self.max_p

        edges = [i for i in range(len(node.loc))]
        chosen_edge = random.choices(edges, a_star_p)

        return chosen_edge[0]


    def generate_antStar_path(self, ant):
        """A* ACO algorithm """
        #start at origin node
        iterator_node = ant.starting_node[0]
        #energy value
        for _ in range(int(len(self.graph.nodes))):
            ant.path.append(iterator_node)
            #if ant encounters a 'dead' node, then break
            if len(iterator_node.adj_list) == 0:
                ant.reset()
                break

            #while not found, explore as much as possible
            #This avoids most of the dead ends
            if not self.found:
                edge_choice = self.get_adjacentNodes(iterator_node)
            else:
                #evaluate the nodes
                edge_choice = self.aStar_distance(iterator_node)

            #add choice to index array, rembering which edge is taken to update the pheromone
            ant.path_index.append(iterator_node.loc[edge_choice])
            #get the next node choice
            node_choice = iterator_node.adj_list[edge_choice]
            current_node_dict = self.graph.__getitem__(iterator_node.node)
            #fitness of path
            edge_fitness = self.evaluate_fitness(current_node_dict[node_choice])
            #add fitness to the ant path
            ant.path_fitness += edge_fitness
            #add the fitness to determin the fitness overtime
            ant.path_length_array.append(edge_fitness)
            #next node choice
            next_node = self.next_nodeObject(iterator_node.loc[edge_choice])
            #check if target
            if next_node.node == self.target_node:
                ant.path.append(next_node)
                ant.path_index.append(iterator_node.loc[edge_choice])
                ant.path_length_array.append(edge_fitness)
                ant.target = True
                self.found = True
                break
            #set the new node to the iterator
            iterator_node = next_node
