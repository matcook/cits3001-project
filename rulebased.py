import cv2
import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym

# Load all templates
mario_templates = [cv2.imread(f'templates/mario{i}.png', cv2.IMREAD_COLOR) for i in ['A', 'B', 'C', 'D', 'E', 'F', 'G']]
blocks_templates = [cv2.imread(f'templates/block{i}.png', cv2.IMREAD_COLOR) for i in range(1, 5)]
koopas_templates = [cv2.imread(f'templates/koopa{i}.png', cv2.IMREAD_COLOR) for i in ['A', 'B']]
mushroom_template = cv2.imread('templates/mushroom_red.png', cv2.IMREAD_COLOR)
pipe_upper_template = cv2.imread('templates/pipe_upper_section.png', cv2.IMREAD_COLOR)
pipe_lower_template = cv2.imread('templates/pipe_lower_section.png', cv2.IMREAD_COLOR)
question_templates = [cv2.imread(f'templates/question{i}.png', cv2.IMREAD_COLOR) for i in ['A', 'B', 'C']]
tall_mario_templates = [cv2.imread(f'templates/tall_mario{i}.png', cv2.IMREAD_COLOR) for i in ['A', 'B', 'C']]
goomba_template = cv2.imread('templates/goomba.png', cv2.IMREAD_COLOR)

def draw_rectangles(image, locations, template_shape, color=(0, 255, 0)):
    """Draw rectangles around all detected positions of an object."""
    # Convert the image to a compatible type
    image = np.ascontiguousarray(image, dtype=np.uint8)
    
    for (x, y) in locations:
        bottom_right = (x + template_shape[1], y + template_shape[0])
        cv2.rectangle(image, (x, y), bottom_right, color, 2)
    
    return image

def detect_objects(observation_bgr, templates):
    best_locations = []

    for template in templates:
        res = cv2.matchTemplate(observation_bgr, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.6
        loc = np.where(res >= threshold)
        
        if loc[0].size:
            best_locations.extend(zip(*loc[::-1]))

    return best_locations

def detect_all_objects(observation):
    observation_bgr = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
    
    return {
        "mario": detect_objects(observation_bgr, mario_templates + tall_mario_templates),
        "goomba": detect_objects(observation_bgr, [goomba_template]),
        "blocks": detect_objects(observation_bgr, blocks_templates),
        "koopas": detect_objects(observation_bgr, koopas_templates),
        "mushroom": detect_objects(observation_bgr, [mushroom_template]),
        "pipe": detect_objects(observation_bgr, [pipe_upper_template] + [pipe_lower_template]),
        #"pipe_lower": detect_objects(observation_bgr, [pipe_lower_template]),
        "question": detect_objects(observation_bgr, question_templates)
    }

def rule_based_action(observation):
    # Make a copy of the observation to draw on
    observation_copy = observation.copy()

    detected_objects = detect_all_objects(observation)
    mario_positions = detected_objects["mario"]
    goomba_positions = detected_objects["goomba"]
    pipe_positions = detected_objects["pipe"]
    question_positions = detected_objects["question"]
    block_positions = detected_objects["blocks"]


    # Drawing rectangles for debugging
    draw_rectangles(observation_copy, mario_positions, mario_templates[0].shape, color=(0, 255, 0))
    draw_rectangles(observation_copy, goomba_positions, goomba_template.shape, color=(0, 0, 255))
    draw_rectangles(observation_copy, pipe_positions, goomba_template.shape, color=(255, 0, 0))
    draw_rectangles(observation_copy, question_positions, goomba_template.shape, color=(255, 0, 255))
    draw_rectangles(observation_copy, block_positions, goomba_template.shape, color=(255, 255, 0))
    # ... (and repeat for other detected objects as necessary with different colors)
    
    # Show the modified observation with rectangles
    cv2.imshow('Debugging Observation', cv2.cvtColor(observation_copy, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)  # Display the image for a short duration

    # Default action
    action = 3  # corresponds to running right in SIMPLE_MOVEMENT

    if mario_positions and goomba_positions:
        mario_central_x = mario_positions[0][0] + mario_templates[0].shape[1] // 2
        for goomba_position in goomba_positions:
            distance = goomba_position[0] - mario_central_x
            if 0 < distance <= 30:
                action = 4  # corresponds to jumping right in SIMPLE_MOVEMENT
                break

    return action

# Setup environment
env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

observation, info = env.reset()

for step in range(10000):
    action = rule_based_action(observation)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()

env.close()
