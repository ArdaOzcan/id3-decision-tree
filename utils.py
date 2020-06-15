import math
import os
from collections import defaultdict

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_COMPLEX


def ratio_of(element, list_):
    """Return the ratio of an element in an iterable.

    Args:
        element (any): Specified element.
        list_ (list): List that contains the specified element.

    Returns:
        float: Ratio of an element in a list.
    """

    count = 0
    for e in list_:
        if e == element:
            count += 1
    return count / len(list_)


def draw_text_center(img, text, center_pos, font_size, thickness):
    """Draw a text to an image whose center is the given position.

    Args:
        img (np.ndarray): Image to be drawn to.
        text (str): Text to be written.
        center_pos (tuple): Center position of the text.
        font_size (float): Size of the font.
        thickness (int): Thickness of the font.
    """

    text = str(text)
    text_size = cv2.getTextSize(text, FONT, font_size, thickness)[0]
    img = cv2.putText(img, text,
                      (int(center_pos[0] - text_size[0] / 2),
                       int(center_pos[1] - text_size[1] / 2)),
                      FONT, font_size, (1, 1, 1), thickness, cv2.LINE_AA)


def count_layers(node, layer=0, layer_counts=None):
    """Count the amount of nodes in each layer and save it to a defaultdict.

    Args:
        node (Node): Current node.
        layer (int, optional): Layer of the table, passed in by its parent. Defaults to 0.
        layer_counts (defaultdict(int), optional): Amount of nodes for each layer. Defaults to None.

    Returns:
        defaultdict(int): Amount of nodes for each layer.
    """

    if layer_counts is None:
        layer_counts = defaultdict(int)

    # Add one to this layer for this node
    layer_counts[layer] += 1

    if node.children == []:
        return

    for c in node.children:
        # Recurse
        count_layers(c, layer=layer+1, layer_counts=layer_counts)

    return layer_counts


def visualize(root):
    """Draw a tree with cv2 representing the tree structure of root.

    Args:
        root (Node): Root node of the tree.

    Returns:
        np.ndarray: Image that has been constructed.
    """

    prompt = 'DRW'
    y_change = root.app.app_config["dimensions"]["yChange"]
    img = np.full((root.app.app_config["dimensions"]["imgHeight"],
                   root.app.app_config["dimensions"]["imgWidth"], 3),
                  .075)

    # Hold how many nodes are in each layer
    # Root is in 0th layer, its children are in 1th...
    layer_counts = count_layers(root)

    # Holds every position that has been drawn to before
    # To prevent overlapping nodes.
    drawn_set = set()

    font_size = root.app.app_config["dimensions"]["fontSize"]
    thickness = root.app.app_config["dimensions"]["thickness"]
    radius = int(img.shape[0]/35)

    def draw(node, x=int(img.shape[1]/2), y=int(img.shape[0]/4), layer=0):
        """Draw nodes in correct positions and write relevant information.

        Args:
            node (Node): Current node that is being drawn. 
            x (int, optional): X position of the current node. Defaults to int(img.shape[1]/2).
            y (int, optional): Y position of the current node. Defaults to int(img.shape[0]/4).
            layer (int, optional): Layer of the current node. Defaults to 0.
        """
        nonlocal img

        total_children = len(node.children)
        for i, c in enumerate(node.children):

            # Divide the width of the image to the
            # amount of nodes in this layer + 1
            # to find the length of every space
            # between nodes in this layer
            child_x = int(x + ((0.5 - i/(total_children-1)) *
                               root.app.app_config["dimensions"]["xChange"] / (layer_counts[layer] + 1)))

            child_y = y + y_change

            # If already drawn to position, increase y until there are no nodes
            while (child_x, child_y) in drawn_set:
                child_y += y_change

            # Convert to int here because of the previous while loop

            child_y = int(child_y)
            drawn_set.add((child_x, child_y))
            img = cv2.line(img, (x, y), (child_x, child_y), (.35, .35, .35))

            # Recurse with child
            draw(c, x=child_x, y=child_y, layer=layer+1)

        # Draw a circle representing this node
        img = cv2.circle(img, (x, y), radius, (.25, .25, .25), -1)

        # Y offset of text
        y_offset = int(img.shape[0]/25)

        # Percentage of possibility.
        ratio = round(ratio_of(node.positive_value, node.result[1]), 4)
        percentage = f'{100 * ratio}%'
        draw_text_center(img, percentage, (x, y), font_size, thickness)

        if node.value is not None:
            draw_text_center(img, node.value,
                             (x, y + y_offset), font_size, thickness)
            # Increase offset so the next text will be lower that this.
            # So if the value was None, next text will be in correct position.
            y_offset += int(img.shape[0]/25)

        if node.split is not None:
            draw_text_center(img, f"{node.split}?",
                             (x, y + y_offset), font_size, thickness)

    # Start recursion
    draw(root)

    # Draw title text
    img = cv2.putText(img,
                      f"Percentage of '{root.positive_value}' in column '{root.result[0]}'",
                      (15, 35), FONT, font_size, (1, 1, 1), thickness, cv2.LINE_AA)

    if root.app.app_config["booleans"]["showImage"]:
        # Show resulting image
        cv2.imshow('Decision tree', img)
        cv2.waitKey(0)

    # Return image for possible writing
    img = cv2.convertScaleAbs(img, alpha=(255.0))
    return img


def entropy(data_list):
    """Calculate and return the entropy of a given list.

    Args:
        data_list (list): List of data whose entropy will be calculated.

    Returns:
        float: Entropy of this list.
    """
    list_length = len(data_list)
    element_amount = defaultdict(int)
    prob_dict = dict()

    # Count amount of occurrence of elements in the list
    for element in data_list:
        element_amount[element] += 1

    def p(x):
        """Calculate or find, and return the probability of an element's occurrence

        If the probability has not been calculated before, calculate it by dividing
        the element amount to the length of the list. Once an element's probability
        is calculated, add it to a dictionary to access it later if the probability 
        of the same element is requested again.

        Args:
            x (any): Element whose probability will be calculated.

        Returns:
            float: Probability of the element's occurrence.
        """
        if x in prob_dict:
            return prob_dict[x]
        prob = element_amount[x] / list_length
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
    """Calculate information gain from given data.

    Args:
        entropies (dict): Dictionary that holds information about entropies of attributes.
        probabilities (dict): Dictionary that holds information about probabilities of attributes.
        result_entropy (float): Entropy of the result column.

    Returns:
        float: Information gain.
    """
    summation = 0
    for attribute in entropies.keys():
        # Information gain formula: https://en.wikipedia.org/wiki/Information_gain_in_decision_trees
        summation += probabilities[attribute] * entropies[attribute]

    return result_entropy - summation
