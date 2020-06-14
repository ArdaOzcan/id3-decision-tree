from colr import Colr as C
import time

def info(msg, prompt="MAIN", highlight_color="ffaa02"):
    '''Log message to console with prompt and color.
    
    Positional arguments:
    msg -- message to be logged
    
    Keyword arguments:
    prompt -- prompted string before the message and time
    highlight_color -- color of prompt, time and highlighted parts of the message
    '''
    
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(C(f"[{prompt}][{current_time}] ", fore=highlight_color, style='bold'), end='')
    
    # Parts that are inside '$' signs are highlighted with the highlight_color
    parts = msg.split('$')
    special = False
    for i, part in enumerate(parts):
        if special:
            print(C(part, fore=highlight_color, style='bold'), end='')
        else:
            print(part, end='')
        special = not special
        
    # Print newline
    print()
    
def error(msg, prompt="MAIN", highlight_color="c22929"): 
    '''Alternative to info with different color.'''
    info(msg, prompt=prompt, highlight_color=highlight_color)
        