import math
from collections import defaultdict

import cv2
import numpy as np


def draw_tree(root):
    '''Draw a tree with cv2 representing the tree structure of root

    Positional arguments:
    root - Root node of the tree
    '''
    # Hold how many nodes are in each layer
    # Root is in 0th layer, its children are in 1th...
    layer_counts = defaultdict(int)

    # Variables
    img = np.full((600, 1000, 3), .075)
    font = cv2.FONT_HERSHEY_COMPLEX
    font_size = .75
    thickness = 2

    def count(table, layer=0):
        '''
        Recursive function to count how many nodes
        are present in each layer and save it to a defaultdict.

        Positional arguments:
        table -- current table

        Keyword arguments:
        layer -- layer of the table, passed in by its parent

        '''
        # Add one to this layer for this table
        layer_counts[layer] += 1

        if table.children == []:
            return

        for c in table.children:
            # Recurse
            count(c, layer=layer+1)

    # Holds every position that has been drawn to before
    # To prevent overlapping nodes.
    drawn_set = set()

    def draw(table, x=int(img.shape[1]/2), y=int(img.shape[0]/4), radius=int(img.shape[0]/35), layer=0):
        '''
        Draw nodes in correct positions and write relevant information.

        Positional arguments:
        table -- current table

        Keyword arguments:
        x -- x position of node, calculated by parent
        y -- y position of node, calculated by parent
        radius -- radius of each node on 2D plane
        layer -- current layer of the node to access layer_counts
        '''
        nonlocal img

        total_children = len(table.children)
        for i, c in enumerate(table.children):
            '''
            Divide the width of the image to the
            amount of nodes in this layer + 1
            to find the length of every space
            between nodes in this layer.
            '''
            child_x = int(x + ((0.5 - i/(total_children-1)) *
                               img.shape[1]/(layer_counts[layer]+1)))
            
            child_y = y + img.shape[0]/4
            
            # If already drawn to position, increase y until there are no nodes
            while (child_x, child_y) in drawn_set:
                child_y += img.shape[0]/8

            # Convert to int here because of the previous while loop
            child_y = int(child_y)
            
            drawn_set.add((child_x, child_y))
            img = cv2.line(img, (x, y), (child_x, child_y), (.35, .35, .35))
            
            # Recurse, base case is no children.
            # If there are no children of this node, it won't be in this loop.
            draw(c, x=child_x, y=child_y, layer=layer+1)

        # Draw a circle representing this node
        img = cv2.circle(img, (x, y), radius, (.25, .25, .25), -1)
        
        # Count values of result column to calculate probabilities
        _, results = table.result
        result_dict = defaultdict(int)
        for r in results:
            result_dict[r] += 1

        # Y offset of text
        y_offset = int(img.shape[0]/25)
        
        # Percentage of possibility.
        text = f'{str(round(100 * result_dict[table.positive_str] / len(results), 2))}%'
        textsize = cv2.getTextSize(text, font, font_size, thickness)[0]
        img = cv2.putText(
            img, text, (x-int(textsize[0]/2), y), font, font_size, (1, 1, 1), thickness, cv2.LINE_AA)

        if table.value is not None:
            text = str(table.value)
            textsize = cv2.getTextSize(text, font, font_size, thickness)[0]
            img = cv2.putText(img, text, (x-int(textsize[0]/2),
                                          y + y_offset), font, font_size, (1, 1, 1), thickness, cv2.LINE_AA)
            
            # Increase offset so the next text will be lower that this.
            # So if the value was None, next text will be in correct position.
            y_offset += int(img.shape[0]/25)

        if table.split is not None:
            text = f"{table.split}?"
            textsize = cv2.getTextSize(text, font, font_size, thickness)[0]
            img = cv2.putText(img, text, (x-int(textsize[0]/2),
                                          y + y_offset), font, font_size, (1, 1, 1), thickness, cv2.LINE_AA)

    # Start recursions
    count(root)
    draw(root)

    # Draw title text
    img = cv2.putText(
        img, f"Percentage of '{root.positive_str}' in column '{root.result[0]}'", (15, 35), font, 1, (1, 1, 1), thickness, cv2.LINE_AA)

    # Show resulting image
    cv2.imshow('Decision tree', img)
    cv2.waitKey(0)


def entropy(data_list):
    '''Calculate entropy of a data list
    
    Positional arguments:
    data_list -- list of data whose entropy will be calculated
    '''
    data_length = len(data_list)
    element_amount = defaultdict(int)
    prob_dict = dict()
    # Count amount of occurrence of elements in the list
    for element in data_list:
        element_amount[element] += 1

    def p(x):
        '''
        Find probability of element x
        If p(x) is not calculated before, store it in a dict to use later
        
        Positional arguments:
        x -- element whose probability will be calculated
        '''
        if x in prob_dict:
            return prob_dict[x]
        prob = element_amount[x] / data_length
        prob_dict[x] = prob
        return prob

    # using set() because 
    # we want to loop through every unique 
    # element once.
    summation = 0
    for element in set(data_list):
        # Shannon entropy formula: https://en.wikipedia.org/wiki/Entropy_(information_theory)
        element_p = p(element)
        summation += -element_p * math.log2(element_p)
    return summation


def info_gain(entropies, probabilities, result_entropy):
    '''Calculate information gain from given data.
    
    Positional arguments:
    entropies -- dictionary that holds entropies of attributes
    probabilities - -dictionary that holds probabilites of attributes
    '''
    summation = 0
    for attribute in entropies.keys():
        # Information gain formula: https://en.wikipedia.org/wiki/Information_gain_in_decision_trees
        summation += probabilities[attribute]*entropies[attribute]
    return result_entropy - summation
