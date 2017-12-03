#!/usr/bin/env python

from sfml import sf
import sfml.graphics as sf_graph
import sfml.system as sf_sys
import numpy as np
import argparse
import cPickle

sf = reload(sf)

WINDOW_W = 400
WINDOW_H = 400


aparser = argparse.ArgumentParser()
aparser.add_argument('map_fname',
                     type=str,
                     help='Path to .npz-file with map to generate tasks for')
args = aparser.parse_args()

cur_map = np.load(args.map_fname)['arr_0']
map_window = sf.RenderWindow(sf.VideoMode(WINDOW_W, WINDOW_H), "MAP")
print(cur_map)
map_squares = [[0] * cur_map.shape[1] for _ in range(cur_map.shape[0])]
map_window.clear(sf_graph.Color.WHITE)
map_window.display()
for row in range(cur_map.shape[0]):
    for col in range(cur_map.shape[1]):
        map_squares[row][col] = sf_graph.RectangleShape((20, 20))
        map_squares[row][col].outline_color = sf.Color.BLACK
        map_squares[row][col].outline_thickness = 1
        # map_squares[row][col].size = sf_sys.Vector2(20, 20)
        if (cur_map[row][col]):
            map_squares[row][col].fill_color = sf_graph.Color.RED
        map_squares[row][col].position = (col * 20, row * 20)
        map_window.draw(map_squares[row][col])
map_window.display()

while map_window.is_open:
    for event in map_window.events:
        if type(event) is sf.CloseEvent:
            map_window.close()
