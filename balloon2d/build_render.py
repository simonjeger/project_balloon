import numpy as np
import pygame, sys

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

def build_render(character, reward_step, reward_epi, world_name, window_size, train_or_test):
    size_x = character.size_x
    size_z = character.size_z

    # general setup
    pygame.init()
    clock = pygame.time.Clock()

    # setting up the main window
    res = int(520/size_z) #40
    screen_width = size_x*res
    screen_height = size_z*res
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('balloon2d')

    # game rectangles (orgin is always on the top left, so increase y to go down)
    start = int(max(character.position[0] - window_size, 0))
    end = int(min(character.position[0] + window_size + 1, size_x))
    size_obs_x = (end - start)*res
    size_obs_y = size_z*2*res

    pos_obs = [int(max(character.position[0] - window_size, 0))*res, 0]
    rec_obs = pygame.Rect(pos_obs[0], pos_obs[1], size_obs_x, size_obs_y)

    path = []
    for i in character.path:
        path.append((i[0]*res, (size_z-i[1])*res))

    size_balloon = size_z/50*res
    pos_balloon = [character.position[0]*res, (size_z - character.position[1])*res]
    rec_balloon = pygame.Rect(pos_balloon[0] - size_balloon/2, pos_balloon[1] - size_balloon/2, size_balloon, size_balloon)

    size_target = size_z/100*res
    pos_target = [character.target[0]*res, (size_z - character.target[1])*res]
    rec_target = pygame.Rect(pos_target[0] - size_target/2, pos_target[1] - size_target/2, size_target, size_target)

    # handling input
    for event in pygame.event.get():
        if event.type == pygame.QUIT: #does user click closing button
            pygame.quit()
            sys.exit()

    # visualize ground detection #remove if not needed anymore
    #distance = character.state[8]
    #bearing = character.state[9]
    #ground_detection = [(pos_balloon[0], pos_balloon[1]), (pos_balloon[0] + distance*np.sin(bearing)*res, pos_balloon[1] + distance*np.cos(bearing)*res)]

    # visualize ceiling
    ceiling = []
    for i in range(len(character.ceiling)):
        pair = (((1+1/len(character.ceiling))*i*res,(size_z - character.ceiling[i])*res))
        ceiling.append(pair)
    ceiling.append((size_x*res, 0))
    ceiling.append((0, 0))

    # colors
    c_background = (34,42,53)
    c_ceiling = (51,63,80,150)
    c_down = (117,21,0,255)
    c_stay = (173,29,0)
    c_up = (237,35,1)
    c_path = (242,242,242)
    c_window = (217,217,217, 50)
    c_target_center = (242,242,242)
    c_target_radius = (217,217,217,50)

    # generate background
    screen.fill(c_background)
    bg = pygame.image.load(yaml_p['data_path'] + train_or_test + '/image/' + world_name + '.png')
    bg = pygame.transform.scale(bg, (screen_width, screen_height))
    screen.blit(bg, (0, 0))

    # draw ceiling
    #pygame.draw.polygon(screen, c_ceiling, ceiling)
    lx, ly = zip(*ceiling)
    min_x, min_y, max_x, max_y = min(lx), min(ly), max(lx), max(ly)
    target_rect = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)
    shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
    pygame.draw.polygon(shape_surf, c_ceiling, [(x - min_x, y - min_y) for x, y in ceiling])
    screen.blit(shape_surf, target_rect)


    # draw transparent observed box on top
    shape_surf = pygame.Surface(pygame.Rect(rec_obs).size, pygame.SRCALPHA)
    pygame.draw.rect(shape_surf, c_window, shape_surf.get_rect())
    screen.blit(shape_surf, rec_obs)

    # visuals
    #pygame.draw.lines(screen, pygame.Color('Black'), False, ground_detection, 1) #remove if not needed anymore
    if len(character.path) > 1:
        pygame.draw.lines(screen, c_path, False, path, 1)
    if character.action == 2:
        pygame.draw.ellipse(screen, c_up, rec_balloon)
    if character.action == 1:
        pygame.draw.ellipse(screen, c_stay, rec_balloon)
    if character.action == 0:
        pygame.draw.ellipse(screen, c_down, rec_balloon)

    # draw target (dense and transparent)
    pygame.draw.ellipse(screen, c_target_center, rec_target)

    radius = yaml_p['radius']*res
    target_rect = pygame.Rect((pos_target), (0, 0)).inflate((radius * 2, radius * 2))
    shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
    pygame.draw.circle(shape_surf, c_target_radius, (radius, radius), radius)
    screen.blit(shape_surf, target_rect)

    # text
    myfont = pygame.font.SysFont('Arial', 15, bold = True)
    t_reward_step = myfont.render('reward_step: ' + str(np.round(reward_step,3)), False, pygame.Color('LightGray'))
    t_reward_epi = myfont.render('reward_epi: ' + str(np.round(reward_epi,3)), False, pygame.Color('LightGray'))
    t_residual = myfont.render('residual: ' + str(np.round(character.state[0:2]*yaml_p['unit'],4)), False, pygame.Color('LightGray'))
    if yaml_p['physics']:
        t_velocity = myfont.render('velocity: ' + str(np.round(character.state[2:4]*yaml_p['unit'],1)), False, pygame.Color('LightGray'))
        t_border = myfont.render('border: ' + str(np.round(character.state[4:8]*yaml_p['unit'],1)), False, pygame.Color('LightGray'))
        t_world_compressed = myfont.render('world_compressed: ' + str(np.round(character.state[8:],1)), False, pygame.Color('LightGray'))
    else:
        t_border = myfont.render('border: ' + str(np.round(character.state[2:6],1)), False, pygame.Color('LightGray'))
        t_world_compressed = myfont.render('world_compressed: ' + str(np.round(character.state[6:],1)), False, pygame.Color('LightGray'))


    screen.blit(t_reward_step,(50,10))
    screen.blit(t_reward_epi,(50,25))
    screen.blit(t_residual,(50,55))
    if yaml_p['physics']:
        screen.blit(t_velocity,(50,70))
    screen.blit(t_border,(50,85))
    screen.blit(t_world_compressed,(50,100))

    # updating the window
    pygame.display.flip()
    #clock.tick(10) #cycles per second
    clock.tick(10)
