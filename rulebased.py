import cv2
import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym

# Load all Mario templates
template_files = [f'templates/mario{chr(i)}.png' for i in range(65, 72)]  # 65 = 'A', 72 = 'G' + 1
mario_templates = [cv2.imread(filename, cv2.IMREAD_COLOR) for filename in template_files]
goomba_template = cv2.imread('templates/goomba.png', cv2.IMREAD_COLOR)

print(SIMPLE_MOVEMENT)

def detect_mario(observation_bgr, templates):
    max_val = -1  # initial score
    best_location = None
    
    for template in templates:
        res = cv2.matchTemplate(observation_bgr, template, cv2.TM_CCOEFF_NORMED)
        
        # Find the maximum matching value and its location for this template
        min_val, template_max_val, min_loc, template_max_loc = cv2.minMaxLoc(res)
        
        # If this template's score is better than the previous best, update the best_location and max_val
        if template_max_val > max_val:
            max_val = template_max_val
            best_location = template_max_loc

    return best_location

def detect_goomba(observation_bgr, template):
    res = cv2.matchTemplate(observation_bgr, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.6  # Adjust this value if needed
    loc = np.where(res >= threshold)
    return list(zip(*loc[::-1]))

def rule_based_action(observation):
    observation_bgr = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
    mario_position = detect_mario(observation_bgr, mario_templates)
    goomba_positions = detect_goomba(observation_bgr, goomba_template)

    # Debug - Visualize detections
    #debug_image = observation_bgr.copy()
    #if mario_position:
    #    mario_end = (mario_position[0] + mario_templates[0].shape[1], mario_position[1] + mario_templates[0].shape[0])
    #    cv2.rectangle(debug_image, mario_position, mario_end, (255, 0, 0), 2)  # Blue rectangle for Mario
    #for pos in goomba_positions:
    #    goomba_end = (pos[0] + goomba_template.shape[1], pos[1] + goomba_template.shape[0])
    #    cv2.rectangle(debug_image, pos, goomba_end, (0, 0, 255), 2)  # Red rectangle for Goombas
    #cv2.imshow('Debug', debug_image)
    #cv2.waitKey(1)

    # Debug - Print positions and distances
    #if mario_position:
    #    print("Mario Position:", mario_position)
    #print("Goomba Positions:", goomba_positions)
    
    # Default to moving right
    action = 3 # 1 corresponds to moving right in SIMPLE_MOVEMENT

    if mario_position and goomba_positions:
        mario_central_x = mario_position[0] + mario_templates[0].shape[1] // 2
        for goomba_position in goomba_positions:
            distance = goomba_position[0] - mario_central_x
            #print("Distance to Goomba:", distance)
            if 0 < distance <= 30:
                action = 4  # 4 corresponds to jumping right in SIMPLE_MOVEMENT
                break
    #print("selected action: ", action)
    return action

# Setup environment
env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

observation, info = env.reset()

for step in range(10000):
    action = rule_based_action(observation)
    obs, reward, terminated, truncated, info = env.step(action)
    #observation, reward, done, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
