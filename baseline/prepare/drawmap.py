#!/usr/bin/env python

from sfml import sf
import sfml.graphics as sf_graph
import sfml.system as sf_sys
import sfml.window as sf_win
import numpy as np
import argparse
import cPickle
import hashlib
import os

sf = reload(sf)

WINDOW_W = 400
WINDOW_H = 400
MAP_SIZE = 20


map_window = sf.RenderWindow(sf.VideoMode(WINDOW_W, WINDOW_H), "MAP")
map_squares = [[0] * MAP_SIZE for _ in range(MAP_SIZE)]
cur_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=int)
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
        if type(event) is sf_win.MouseButtonEvent:
            if event.pressed:
                row = event.position.y // 20
                col = event.position.x // 20
                if (cur_map[row][col] == 0):
                    cur_map[row][col] = 1
                    map_squares[row][col].fill_color = sf_graph.Color.RED
                else:
                    cur_map[row][col] = 0
                    map_squares[row][col].fill_color = sf_graph.Color.WHITE
                map_window.draw(map_squares[row][col])
                map_window.display()

print(cur_map)
print("Save y/N?")
if (raw_input() == 'y'):
    map_id = hashlib.md5(cur_map.tostring()).hexdigest()
    map_fname = os.path.join(map_id)
    if not os.path.exists(map_fname + '.npz'):
        np.savez_compressed(map_fname, cur_map)
