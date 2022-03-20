from django.shortcuts import render
import folium
import openrouteservice as ors
from folium.plugins import MarkerCluster
from folium.plugins import MousePosition
from django.http import HttpResponse
from django.http import JsonResponse

import osmnx as ox
import networkx as nx
import numpy as np
import random
from IPython.display import IFrame, display
import matplotlib.pyplot as plt
import time
import folium
import webbrowser





ors_key = '5b3ce3597851110001cf624844229b68a7af4311b243a28407590a15'


def render_location(client, l):
    # geocode
    try:
        geocode = client.pelias_search(text=l)
    except client.HTTPError(400):
        print("error")

    return(geocode)


def index(request):
    #client
    client = ors.Client(key=ors_key)
    #
    location_cds = []
    #map
    m = folium.Map(location=[50.73207170509859,-3.5170272043617445,],
                    tiles='cartodbpositron',
                    zoom_start=5,
                    max_bounds=True)

    pip = None

    if request.method == 'POST':
        l = request.POST.get('location')
        r = request.POST.get('range_search')
        geocode = render_location(client, l)
        for result in geocode['features']:
            folium.Marker(location=list(reversed(result['geometry']['coordinates'])),
            icon=folium.Icon(icon='building', color='green', prefix='fa'),
            popup=folium.Popup(result['properties']['name']), draggable=True).add_to(m)


        pip = geocode['features'][0]['geometry']['coordinates']

        isochrone = client.isochrones(locations=[pip], range_type='distance', range=[int(r)], attributes=['area'])
        folium.GeoJson(isochrone, name='isochrone').add_to(m)

        folium.LayerControl().add_to(m)

        m = m._repr_html_()

        context={
            'display_map' : m
        }

        return render(request, 'index.html', context)

    # if request.method == 'GET':
    #     r = request.GET.get('range_search')
    #     print(r)

        # isochrone = client.isochrones(locations=[new_location_cds], range_type='distance', range=[int(r)], attributes=['area'])
        # folium.GeoJson(isochrone, name='isochrone').add_to(m)


    #isochrone area
    #isochrone = client.isochrones(locations=coordinates,range_type='distance', range=[100,200], attributes=['area'])

    #folium.GeoJson(isochrone, name='isochrone').add_to(m)

    # popup_message = 'Area in 100m and 200m'
    #
    # folium.Marker(location_cds, popup=popup_message, tooltip='Open').add_to(m)
    #
    # folium.LayerControl().add_to(m)


    #folium.LatLngPopup().add_to(m)


    folium.ClickForMarker().add_to(m)

    m = m._repr_html_()

    context={
        'display_map' : m
    }

    return render(request, 'index.html', context)

class PATH():
    path = []
    running = False
    done = False

    def reset():
        path = []
        running = False
        done = False

def d_first_search_control(E):
    found_ant = E.initial_search()
    # for i in found_ant.visited_nodes:
    #     print(i.node)
    #     print(i.adj_list)
    PATH.path = found_ant.xy_data
    PATH.running = True
    time.sleep(1)
    PATH.done = True

def aco_not_show(E):
    i = 0

    while i < E.fitness_evaluations:
        return_path = E.ant_optimisation()
        print(i)
        PATH.path = return_path
        i+=1

    PATH.path = return_path
    PATH.running = True
    time.sleep(1)
    PATH.done = True


def aco_control(E):
    i = 0
    #PATH.running = True
    while i < E.fitness_evaluations:
        if PATH.running == False:
            return_path = E.ant_optimisation()
            print(i)
            PATH.path = return_path
            PATH.running = True
            i+=1
        else:
            time.sleep(0.05)


    PATH.path = return_path
    #PATH.running = True
    time.sleep(1)
    PATH.done = True
    #print("#########", PATH.path)


def vis_logic(request):
    print("Starting visual")

    PATH.reset()


    origin = [51.530249, -3.177608]
    destination = [51.52494521411801, -3.1640268228242125]
    dest2 = [51.513793, -3.154131]
    dest3 = [51.503790, -3.166346]
    p_inp = 0.7
    e_inp = 0.9
    bbox =  [[-3.190215, 51.51838], [-3.149435, 51.537751]]
    bbox2 = [[-3.209977, 51.502882], [-3.127302, 51.540775]]
    E = Environment(bbox2, origin, dest3, p_inp, e_inp)
    #aco_control(E)
    #aco_not_show(E)
    d_first_search_control(E)

    return HttpResponse('Success')



def visualisation(request):
    #'bbox' : [[51.51838, -3.190215,], [51.537751, -3.149435]],
    #'bbox' :[[51.502882, -3.209977], [51.540775, -3.127302]],
    context={

            'bbox' :[[51.502882, -3.209977], [51.540775, -3.127302]],
            'origin': [51.530249, -3.177608],
            'dest': [51.503790, -3.166346]
        }

    return render(request, 'visual.html', context)

def vis_finish(request):
    PATH.running = False
    return HttpResponse('Success')

def points_api(request):
    return JsonResponse({'array': PATH.path, 'running' : PATH.running, 'done': PATH.done})


def grouped(iterable, n):
    a = zip(*[iter(iterable)]*n)
    return a


def simplfy(array):
    array = grouped(array, 2)
    sort_array = []
    for x, y in array:
        temp = []
        temp.append(float(x))
        temp.append(float(y))
        sort_array.append(temp)

    return sort_array


def locations(request):
    array = request.GET.getlist("locationArray[]")
    main_loc_array = simplfy(array)
    logic(main_loc_array)

    return HttpResponse('Success')



def logic(main_loc_array):
    print(main_loc_array)

    point_object_array = []


    for i in main_loc_array:
        p = Pointobj(i)
        point_object_array.append(p)

    E = EnvCreate()
    E.get_isochrones(point_object_array)



class Pointobj():
    def __init__(self, point):
        self.point = point
        self.isochrone_bounds = []
        self.possible_locations = []
        self.distance_m = []


class EnvCreate():
    def __init__(self):
        self.client = ors.Client(key=ors_key)
        self.points_array = []
        self.boundingBox = []

    def get_isochrones(self, loc_array):
        for point in loc_array:
            isochrone = self.client.isochrones(locations=[point.point],range_type='distance', range=[200], attributes=['area'])
            point.isochrone_bounds = isochrone['features'][0]['geometry']['coordinates'][0]

        self.bounds(loc_array)
        #return loc_array

    def generate_paths(self, loc_array):
        for point in loc_array:
            p = Point(point)
            self.points_array.append(p)

        a = self.getDistances(p, loc_array)

    def getDistances(self, p, loc_array):
        temp_locations = []
        for l_point in loc_array:
            if l_point != p.point:
                temp_locations.append(l_point)

        p.possible_locations = temp_locations
        print("current ", p.point)
        print("Possible ",temp_locations )

        result_q = self.client.distance_matrix(locations = loc_array, profile='driving-car',
        metrics = ['distance'],
        units= 'm')
        print(result_q)

    def bounds(self, points):
        if len(points) == 0:
            raise ValueError("Can't compute bounding box of empty list")

        self.minx, self.miny = float("inf"), float("inf")
        self.maxx, self.maxy = float("-inf"), float("-inf")

        for node in points:
            listNode = node.isochrone_bounds
            print("list", listNode)
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
                elif y > self.maxy:
                    self.maxy = y

        self.boundingBox.append([self.minx, self.miny])
        self.boundingBox.append([self.maxx, self.maxy])
        print(self.boundingBox)

    def width(self):
        return self.maxx - self.minx

    def height(self):
        return self.maxy - self.miny


class ANT():
    def __init__(self, ant_number):
        self.n = ant_number
        self.visited_nodes = []
        self.path = []
        self.path_index = []
        self.xy_data = []
        self.path_fitness = 0
        self.starting_node = None
        self.target = False

    def reset(self):
        self.visited_nodes = []
        self.path = []
        self.path_index = []
        self.path_fitness = 0
        self.target = False
        self.xy_data = []




class Node():
    def __init__(self, node_number):
        self.node = node_number
        self.adj_list = []
        self.loc = []
        self.p = []
        self.seen = False


class Environment():
    def __init__(self, bounding_box, origin_loc, destination_loc , p_inp, e_inp):
        self.vis_logic = True
        self.vis_running = False
        self.fitness_evaluations = 10
        self.total_graph_length = 0
        self.bbox = bounding_box
        self.graph = None
        self.gdf_nodes = None
        self.nodeObjectList = []
        self.antObjectList = []
        #//
        self.node_list = []
        self.adj_matrix = [[]]
        self.p_val = p_inp
        self.e_val = e_inp
        #//
        self.target_node = destination_loc
        self.start_node = origin_loc
        self.found = False
        #//
        self.best_path = [[]]
        self.seen_nodes = []
        #//
        self.area()
        self.ant_innit()
        self.map_fitness()
        #self.main_run()

    def ant_innit(self):
        s_node = self.set_targets()
        for i in range(100):
            ant = ANT(i)
            ant.starting_node = s_node
            self.antObjectList.append(ant)

#         for i in self.nodeObjectList:
#             print(i.node, i.adj_list, i.loc, i.p)


    def area(self):
        north, south, east, west = self.bbox[1][1], self.bbox[0][1], self.bbox[0][0],  self.bbox[1][0]
        self.graph = ox.graph_from_bbox(north, south, east, west, network_type='drive')
        self.node_list = list(self.graph)
        self.adj_matrix = nx.adjacency_matrix(self.graph, nodelist=None, dtype=None, weight='weight')
        self.adj_matrix = self.adj_matrix.toarray(order=None, out=None)
        self.gen_nodes()


    def set_targets(self):
        self.start_node = ox.distance.nearest_nodes(self.graph, self.start_node[1], self.start_node[0])
        self.target_node = ox.distance.nearest_nodes(self.graph, self.target_node[1], self.target_node[0])

        c = 0
        for node in self.nodeObjectList:
            if self.start_node == node.node:
                return [node,c]
            c+=1

    def map_fitness(self):
        for node in self.nodeObjectList:
            node_dict = self.graph.__getitem__(node.node)
            for adj in node.adj_list:
                self.total_graph_length += node_dict[adj][0]['length']

        print("Total fit:", self.total_graph_length)


    def links(self, array):
#         for link in array:
#             if link !=0:
        location = np.where(array != 0)
        for list_l in location:
            return list_l


    def neighbours(self, node_number):
        adjacent_nodes = self.graph.__getitem__(node_number)
        pos_arr = list(adjacent_nodes.keys())
        return np.sort(pos_arr)

    def pheromone(self, adj_array):
        p_array = []
        for edge in adj_array:
            p_array.append(random.uniform(0.4,0.7))
        return p_array

    def gen_nodes(self):
        self.nodeObjectList = []

        c = 0
        for n in self.node_list:
            node = Node(n)
            arr = self.adj_matrix[c]
            loc = self.links(arr)
            node.loc = loc
            adj_n = self.neighbours(n)
            node.adj_list = adj_n
            pher = self.pheromone(adj_n)
            node.p = pher
            self.nodeObjectList.append(node)
            c+=1


#         self.gdf_nodes, _ = ox.graph_to_gdfs(self.graph)
#         print(self.gdf_nodes["osmid"])

    def initial_search(self):
        #gdf_nodes, gdf_edges = ox.graph_to_gdfs(self.graph,nodes=True, edges=True,node_geometry=True,fill_edge_geometry=True)
        print("uasjfhsbf")
        for ant in self.antObjectList:
            self.find_init_path(ant)
            if self.found:
                print("FOUND")
                self.set_xy_data(ant)
                return ant

        self.set_xy_data(self.antObjectList[0])

        return self.antObjectList[0]

    def main_run(self):

        fitness_evaluations = 3 # 100
        i = 0

        while i < fitness_evaluations:
            return_path = self.ant_optimisation()
            i+=1
        self.show_graph(return_path)


    def show_graph(self, path):

        print("PP", path)

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

    def ant_optimisation(self):
        for ant in self.antObjectList:
            self.generate_ant_path(ant)
        fit_ant = self.best_fitness()
        best_path = self.gen_ant_xy(fit_ant)
        self.graph_update(fit_ant)
        self.evaporate_pheromone()
        # best_path = [n.node for n in fit_ant.path]
        self.reset()
        return best_path

    def reset(self):
        for ant in self.antObjectList:
            ant.visited_nodes = []
            ant.path = []
            ant.path_index = []
            ant.path_fitness = 0
            ant.target = False
            ant.xy_data = []

    def evaporate_pheromone(self):
        for node in self.nodeObjectList:
            temp_p = []
            for pher in node.p:
                if pher*self.e_val < 0.05:
                    temp_p.append(0.05)
                else:
                    temp_p.append(pher*self.e_val)
            node.p = temp_p

    def graph_update(self, ant_obj):
        print(self.total_graph_length)
        print(ant_obj.path_fitness)
        fitness_update = (self.total_graph_length/ant_obj.path_fitness)/100
        print(fitness_update)
        # fitness_update = 100/ant_obj.path_fitness
        if ant_obj.target:
            fitness_update += 0.1
            node_iter = ant_obj.starting_node[1]

            init_n = self.nodeObjectList[node_iter]

            i = 0
            for node in ant_obj.path:
                choice = ant_obj.path_index[i]
                adj_c = 0
                for adj in node.loc:
                    if adj == choice:
                        if node.p[adj_c] + fitness_update > 1:
                            node.p[adj_c] = 0.91
                        else:
                            node.p[adj_c] += fitness_update
                    adj_c +=1
                i+=1

    def gen_ant_xy(self, ant):
        for i in range(len(ant.path)-1):
            current_node_dict = self.graph.__getitem__(ant.path[i].node)
            coords = self.get_lineObject(current_node_dict[ant.path[i+1].node], ant.path[i+1].node)
            for j in coords:
                ant.xy_data.append(j)

        return ant.xy_data


    def set_xy_data(self, ant):

        fit = 0
        list_nnn = []
        for cc in range(len(ant.visited_nodes)-1):
            if ant.visited_nodes[cc].node != ant.visited_nodes[cc+1].node:
                list_nnn.append(ant.visited_nodes[cc])
                cc+=1
            else:
                cc+=1

        list_nnn.append(ant.visited_nodes[cc])
        print("################",list_nnn)
        ant.xy_data = []
        for i in range(len(list_nnn)-1):
            current_node_dict = self.graph.__getitem__(list_nnn[i].node)
            coords = self.get_lineObject(current_node_dict[list_nnn[i+1].node], list_nnn[i+1].node)
            fit += current_node_dict[list_nnn[i+1].node][0]['length']
            for j in coords:
                ant.xy_data.append(j)

    def best_fitness(self):

        cost = self.total_graph_length
        best_ant = None

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

    def evaluate_fitness(self, node_dict, ant):
        edge_cost = node_dict[0]['length']
        return edge_cost

    def next_nodeObject(self, index):
        return self.nodeObjectList[index]

    def get_adjacentNodes(self, node):

        # t_node = self.graph.nodes[self.target_node]
        # x2 = t_node["x"]
        # y2 = t_node["y"]

        # el = 100
        #
        #
        # j = 0
        # n_d = 0
        # for i in node.adj_list:
        #     info = self.graph.nodes[i]
        #     x1 = info["x"]
        #     y1 = info["y"]
        #     euclidean_dist_vec = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        #     if euclidean_dist_vec < el:
        #         el = euclidean_dist_vec
        #         n_d = np.where(node.adj_list == i)
        #         n_d = n_d[0][0]
        #     j+=1
        #
        # node.p[n_d] + 0.05
        # if node.p[n_d] > 1:
        #     node.p[n_d] = 0.91

        edges = [i for i in range(len(node.loc))]

        chosen_edge = random.choices(edges, node.p)

        return chosen_edge[0]

    def get_lineObject(self, node_dict, node):
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
        return coords


    def get_initadjacentNodes(self, node):
        for j in node.loc:
            if not (self.nodeObjectList[j].seen):
                r = np.where(node.loc == j)
                return r[0][0]
        return None

    def find_init_path(self, ant):
        iterator_node = ant.starting_node[0]
        ant.visited_nodes.append(iterator_node)
        #iterator_node.seen = True
        while not self.found:
            edge_choice = self.get_initadjacentNodes(iterator_node)
            if edge_choice == None:
                ant.visited_nodes.pop()
                if len(ant.visited_nodes) == 0:
                    break
                last = ant.visited_nodes[-1]
                iterator_node = last
            else:
                next_node = self.next_nodeObject(iterator_node.loc[edge_choice])
                ant.visited_nodes.append(next_node)
                next_node.seen = True
                iterator_node = next_node
                if next_node.node == self.target_node:
                    ant.visited_nodes.append(next_node)
                    self.found = True
                    break


    def generate_ant_path(self, ant):

        iterator_node = ant.starting_node[0]
        for _ in range(int(len(self.graph.nodes)/2)):
            ant.path.append(iterator_node)
            if len(iterator_node.adj_list) == 0:
                ant.reset()
                break
            edge_choice = self.get_adjacentNodes(iterator_node)
            ant.path_index.append(iterator_node.loc[edge_choice])
            node_choice = iterator_node.adj_list[edge_choice]
            current_node_dict = self.graph.__getitem__(iterator_node.node)
            edge_fitness = self.evaluate_fitness(current_node_dict[node_choice], ant)
            ant.path_fitness += edge_fitness
            next_node = self.next_nodeObject(iterator_node.loc[edge_choice])
            if next_node.node == self.target_node:
                ant.path.append(next_node)
                ant.path_index.append(iterator_node.loc[edge_choice])
                ant.target = True
                break
            iterator_node = next_node
