import time
from collections import defaultdict

import utils
from node import Node
from logger import info

if __name__ == '__main__':
    info("Program started.")
    root = Node()
    # If successful
    if root.load_from_csv('data/play_data.csv', 'Yes'):
        root.create_decision_tree_id3()
        utils.draw_tree(root)
