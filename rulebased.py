import cv2
import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym

print(SIMPLE_MOVEMENT)

# Load all templates
mario_templates = [cv2.imread(f'templates/mario{i}.png', cv2.IMREAD_COLOR) for i in ['A', 'B', 'C', 'D', 'E', 'F', 'G']]
blocks_templates = [cv2.imread(f'templates/block{i}.png', cv2.IMREAD_COLOR) for i in range(1, 5)]
koopas_templates = [cv2.imread(f'templates/koopa{i}.png', cv2.IMREAD_COLOR) for i in ['A', 'B']]
#mushroom_template = cv2.imread('templates/mushroom_red.png', cv2.IMREAD_COLOR)
pipe_upper_template = cv2.imread('templates/pipe_upper_section.png', cv2.IMREAD_COLOR)
pipe_lower_template = cv2.imread('templates/pipe_lower_section.png', cv2.IMREAD_COLOR)
#question_templates = [cv2.imread(f'templates/question{i}.png', cv2.IMREAD_COLOR) for i in ['A', 'B', 'C']]
#tall_mario_templates = [cv2.imread(f'templates/tall_mario{i}.png', cv2.IMREAD_COLOR) for i in ['A', 'B', 'C']]
goomba_template = cv2.imread('templates/goomba.png', cv2.IMREAD_COLOR)

def detect_objects(observation_bgr, templates, roi=None):
    best_locations = []

    if roi:  # If a Region of Interest (ROI) is provided
        x_start, x_end, y_start, y_end = roi
        observation_bgr = observation_bgr[y_start:y_end, x_start:x_end]

    for template in templates:
        res = cv2.matchTemplate(observation_bgr, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.46
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
    #mario_positions = detect_objects(observation_bgr, mario_templates + tall_mario_templates)
    mario_positions = detect_objects(observation_bgr, mario_templates)
    
    # Calculate ROI based on Mario's position
    if mario_positions:
        mario_x, mario_y = mario_positions[0]
        mario_width, mario_height = mario_templates[0].shape[1], mario_templates[0].shape[0]
        x_start, x_end = mario_x + mario_width, min(mario_x + mario_width + 100, observation_bgr.shape[1])
        y_start, y_end = 0, observation_bgr.shape[0]  # Keeping the full height for simplicity
        roi = (x_start, x_end, y_start, y_end)
    else:
        roi = None
    
    return {
        "mario": mario_positions,
        "goomba": detect_objects(observation_bgr, [goomba_template], roi),
        "blocks": detect_objects(observation_bgr, blocks_templates, roi),
        "koopas": detect_objects(observation_bgr, koopas_templates, roi),
        #"mushroom": detect_objects(observation_bgr, [mushroom_template], roi),
        "pipe_upper": detect_objects(observation_bgr, [pipe_upper_template], roi),
        "pipe_lower": detect_objects(observation_bgr, [pipe_lower_template], roi),
        #"question": detect_objects(observation_bgr, question_templates, roi)
    }

def rule_based_action(observation):
    detected_objects = detect_all_objects(observation)
    mario_positions = detected_objects["mario"]
    goomba_positions = detected_objects["goomba"]
    pipe_upper_positions = detected_objects["pipe_upper"]

    # Default action
    action = 1  # corresponds to running right in SIMPLE_MOVEMENT

    if mario_positions and goomba_positions:
        mario_central_x = mario_positions[0][0] + mario_templates[0].shape[1] // 2
        for goomba_position in goomba_positions:
            distance = goomba_position[0] - mario_central_x
            if 0 < distance <= 30:
                action = 4  # corresponds to jumping right in SIMPLE_MOVEMENT
                break

    if mario_positions and pipe_upper_positions:
        mario_central_x = mario_positions[0][0] + mario_templates[0].shape[1] // 2
        for pipe_position in pipe_upper_positions:
            distance = pipe_position[0] - mario_central_x
            if 0 < distance <= 45:
                action = 4  # corresponds to jumping right in SIMPLE_MOVEMENT
                break
    
    return action

# Setup environment
env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

observation, info = env.reset()

for step in range(10000):
    action = rule_based_action(observation)
    #print(action)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()

env.close()
