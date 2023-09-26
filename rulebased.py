import cv2
import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym

# Actions include
# 0. None
# 1. Right
# 2. Right jump
# 3. Right sprint
# 4. Right sprint jump
# 5. Jump
# 6. Left
print(SIMPLE_MOVEMENT)

##################
# Initialisation #
##################

# Constants for area of screen to look in
HORIZONTAL_DISTANCE = 60
VERTICAL_DISTANCE = 120

# Image matching thresholds
MARIO_THRESHOLD = 0.46
GOOMBA_THRESHOLD = 0.6
PIPE_THRESHOLD = 0.7
KOOPA_THRESHOLD = 0.7
BLOCK_THRESHOLD = 0.8

# Load templates
mario_templates = [cv2.imread(f'templates/mario{i}.png', cv2.IMREAD_COLOR) for i in ['A', 'B', 'C', 'D', 'E', 'F', 'G']]
ground_block_template = cv2.imread('templates/block2.png', cv2.IMREAD_COLOR)
step_block_template = cv2.imread('templates/block4.png', cv2.IMREAD_COLOR)
koopas_templates = [cv2.imread(f'templates/koopa{i}.png', cv2.IMREAD_COLOR) for i in ['A', 'B','C', 'D']]
pipe_upper_template = cv2.imread('templates/pipe_upper_section.png', cv2.IMREAD_COLOR)
goomba_template = cv2.imread('templates/goomba.png', cv2.IMREAD_COLOR)

# Keeps track of last position of ground blocks
# Used to get agent unstuck if hasn't moved after period of time
last_ground_block_positions = []
last_ground_block_positions_timer = 0

####################
# Object detection #
####################

def detect_objects(observation_bgr, templates, threshold, roi=None):
    best_locations = []

    if roi:  # If a Region of Interest (ROI) is provided
        x_start, x_end, y_start, y_end = roi
        observation_bgr = observation_bgr[y_start:y_end, x_start:x_end]

    for template in templates:
        if observation_bgr.shape[0] < template.shape[0] or observation_bgr.shape[1] < template.shape[1]:
            continue  # Skip this template
        res = cv2.matchTemplate(observation_bgr, template, cv2.TM_CCOEFF_NORMED)
        threshold = threshold
        loc = np.where(res >= threshold)
        
        if loc[0].size:
            if roi:  # Adjust the location based on the ROI
                adjusted_loc = [x + x_start for x in loc[1]], [y + y_start for y in loc[0]]
                best_locations.extend(zip(*adjusted_loc))
            else:
                best_locations.extend(zip(*loc[::-1]))

    return best_locations

def detect_all_objects(observation):
    observation_bgr = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
    mario_positions = detect_objects(observation_bgr, mario_templates, MARIO_THRESHOLD)
    if mario_positions:
        y_end = mario_positions[0][1] + 70
        x_start = mario_positions[0][0] + 5
    else:
        y_end = observation_bgr.shape[0]
        x_start = observation_bgr.shape[1] // 2
    x_end = HORIZONTAL_DISTANCE + observation_bgr.shape[1] // 2
    y_start = VERTICAL_DISTANCE 
    roi = (x_start, x_end, y_start, y_end)
    
    return {
        "mario": mario_positions,
        "goomba": detect_objects(observation_bgr, [goomba_template], GOOMBA_THRESHOLD, roi),
        "ground_block": detect_objects(observation_bgr, [ground_block_template], BLOCK_THRESHOLD, roi),
        "step_block": detect_objects(observation_bgr, [step_block_template], BLOCK_THRESHOLD, roi),
        "koopas": detect_objects(observation_bgr, koopas_templates, KOOPA_THRESHOLD, roi),
        "pipe_upper": detect_objects(observation_bgr, [pipe_upper_template], PIPE_THRESHOLD, roi),
    }

########################
# Debug window drawing #
########################

def draw_borders_on_detected_objects(observation, detected_objects):
    # Highlighting colours in BGR format
    color_dict = {
        "mario": (0, 0, 255),
        "goomba": (0, 255, 0),
        "ground_block": (255, 0, 0),
        "step_block": (255, 0, 0),
        "koopas": (255, 255, 0),
        "pipe_upper": (255, 0, 255),
    }

    # Iterate through object types
    for obj_type, positions in detected_objects.items():
        for position in positions:
            top_left = position

            # Try to find the template variable using both singular and plural form
            template_name_singular = f"{obj_type}_template"
            template_name_plural = f"{obj_type}_templates"

            # Get template from global list of templates
            template = globals().get(template_name_singular, 
                                     globals().get(template_name_plural, [None])[0])
            
            # No valid template
            if template is None:
                continue
            
            # Calculate bottom right point of image
            bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])

            # Draw rectangle over object
            cv2.rectangle(observation, top_left, bottom_right, color_dict.get(obj_type, (0, 0, 255)), 2)  # 2 is the thickness of the rectangle border
    
    # Observation now has game state with borders drawn over detected object
    return observation

####################
# Game/Agent rules #
####################

def rule_based_action(observation):
    global last_ground_block_positions
    global last_ground_block_positions_timer

    # Separate detected objects
    detected_objects = detect_all_objects(observation)
    mario_positions = detected_objects["mario"]
    goomba_positions = detected_objects["goomba"]
    pipe_upper_positions = detected_objects["pipe_upper"]
    ground_block_positions = detected_objects["ground_block"]
    step_block_positions = detected_objects["step_block"]
    koopa_positions = detected_objects["koopas"]

    # Default action
    action = 1

    # Get centre of agent's detected x position
    if mario_positions:
        mario_central_x = mario_positions[0][0] + mario_templates[0].shape[1] // 2
    else:
        mario_central_x = 0

    # Jump over Goombas
    if goomba_positions:
        for goomba_position in goomba_positions:
            distance = goomba_position[0] - mario_central_x
            if 0 < distance <= 20:
                action = 4
                break

    # Jump over Koopas
    if koopa_positions:
        for koopa_position in koopa_positions:
            distance = koopa_position[0] - mario_central_x
            if 0 < distance <= 25:
                action = 4
                break
    
    # Jump over pipes
    if pipe_upper_positions:
        for pipe_position in pipe_upper_positions:
            distance = pipe_position[0] - mario_central_x
            if 0 < distance <= 45:
                action = 4
                break

    # Jump up steps
    if step_block_positions:
        for step_position in step_block_positions:
            distance = step_position[0] - mario_central_x
            if distance <= 15:
                action = 4
                break
    
    # Jump over gaps
    if len(ground_block_positions) < 4:
        action = 4
    
    # Reset action to sprint right if in top part of screen
    # Used to jump over gaps
    if mario_positions: 
        if mario_positions[0][1] < 126:
            action = 3
        elif mario_positions[0][1] > 126 and step_block_positions:
            action = 4  

    # If still in same place, then likely stuck on something, so increment timer
    if last_ground_block_positions == ground_block_positions:
        last_ground_block_positions_timer += 1
    else:
        last_ground_block_positions_timer = 0

    # Move left, then sprint right, then jump right
    # Based on timer values
    if last_ground_block_positions_timer in range(50, 55):
        action = 6

    if last_ground_block_positions_timer in range(55, 65):
        action = 3

    if last_ground_block_positions_timer in range(65, 75):
        action = 2

    # Reset timer
    if last_ground_block_positions_timer > 75:
        last_ground_block_positions_timer = 0
    
    last_ground_block_positions = ground_block_positions

    return action, detected_objects

###################
# Run environment #
###################

# Setup environment
env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

observation, info = env.reset()

# Main gameloop
for step in range(10000):
    # Select action based on observation of game state
    action, detected_objects = rule_based_action(observation)
    
    # Visual debug
    # Draw borders around detected objects
    observation_with_borders = draw_borders_on_detected_objects(observation.copy(), detected_objects)
    cv2.imshow("Debug Observation", cv2.cvtColor(observation_with_borders, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)

    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)

    # Handle game reseting if agent dies
    if terminated or truncated:
        observation, info = env.reset()

env.close()
