import numpy as np
import torch
import pygame, sys

from visualize_world import visualize_world

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
    size_y = character.size_y
    size_z = character.size_z

    # general setup
    pygame.init()
    clock = pygame.time.Clock()

    # setting up the main window
    res = int(250/size_z) #520
    screen_width = size_x*res
    screen_height = (size_y + 2*size_z)*res
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('balloon2d')

    # handling input
    for event in pygame.event.get():
        if event.type == pygame.QUIT: #does user click closing button
            pygame.quit()
            sys.exit()

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

    # fill background
    screen.fill(c_background)

    # generate images
    tensor = torch.load(yaml_p['data_path'] + train_or_test + '/tensor/' + world_name + '.pt')
    visualize_world(tensor, character.position)

    for dim in ['xy', 'xz', 'yz']:
        if dim == 'xy':
            idx_x = 0
            idx_y = 1
            dist_to_top = 2*size_z
            dist_to_bottom = size_y + 2*size_z
            loc_size_x = size_x
            loc_size_y = size_y
        if dim == 'xz':
            idx_x = 0
            idx_y = 2
            dist_to_top = size_z
            dist_to_bottom = 2*size_z
            loc_size_x = size_x
            loc_size_y = size_z
            ceiling_raw = character.ceiling[:, int(character.position[1])]
        if dim == 'yz':
            idx_x = 1
            idx_y = 2
            dist_to_top = 0
            dist_to_bottom = size_z
            loc_size_x = size_y
            loc_size_y = size_z
            ceiling_raw = character.ceiling[int(character.position[0]),:]

        position_x = character.position[idx_x]
        position_y = character.position[idx_y]
        target_x = character.target[idx_x]
        target_y = character.target[idx_y]

        # game rectangles (orgin is always on the top left, so increase y to go down)
        start = int(max(position_x - window_size, 0))
        end = int(min(position_x + window_size + 1, loc_size_x))
        size_obs_x = (end - start)*res

        if dim != 'xy':
            size_obs_y = loc_size_y*res
            #pos_obs = [int(max(position_x - window_size, 0))*res, dist_to_top*res]
            pos_obs = [int(position_x - window_size)*res, dist_to_top*res]
        else:
            size_obs_y = (end - start)*res
            #pos_obs = [int(max(position_x - window_size, 0))*res, int(max(position_y - window_size, 0))*res]
            pos_obs = [int(position_x - window_size)*res, int(dist_to_bottom - position_y - window_size)*res]

        rec_obs = pygame.Rect(pos_obs[0], pos_obs[1], size_obs_x, size_obs_y)

        path = []
        for i in character.path:
            path.append((i[idx_x]*res, (dist_to_bottom-i[idx_y])*res))

        size_balloon = size_z/50*res
        pos_balloon = [position_x*res, (dist_to_bottom - position_y)*res]
        rec_balloon = pygame.Rect(pos_balloon[0] - size_balloon/2, pos_balloon[1] - size_balloon/2, size_balloon, size_balloon)

        size_target = size_z/100*res
        pos_target = [target_x*res, (dist_to_bottom - target_y)*res]
        rec_target = pygame.Rect(pos_target[0] - size_target/2, pos_target[1] - size_target/2, size_target, size_target)

        # visualize ground detection #remove if not needed anymore
        #distance = character.state[8]
        #bearing = character.state[9]
        #ground_detection = [(pos_balloon[0], pos_balloon[1]), (pos_balloon[0] + distance*np.sin(bearing)*res, pos_balloon[1] + distance*np.cos(bearing)*res)]

        # visualize ceiling
        if dim != 'xy':
            ceiling = []
            for i in range(len(ceiling_raw)):
                pair = (((1+1/len(ceiling_raw))*i*res,(dist_to_bottom - ceiling_raw[i])*res))
                ceiling.append(pair)
            ceiling.append((loc_size_x*res, dist_to_top*res))
            ceiling.append((0, dist_to_top*res))

        # generate background
        bg = pygame.image.load('render/wind_map_' + dim + '.png')
        img_height = int(screen_height*loc_size_y/(size_y + 2*size_z))
        img_width = int(img_height*loc_size_x/loc_size_y)
        bg = pygame.transform.scale(bg, (img_width, img_height))
        screen.blit(bg, (0, dist_to_top*res))

        if dim != 'xy':
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
        myfont = pygame.font.SysFont('Arial', 10, bold=False)
        t_reward_step = myfont.render('reward_step: ' + str(np.round(reward_step,3)), False, pygame.Color('LightGray'))
        t_reward_epi = myfont.render('reward_epi: ' + str(np.round(reward_epi,3)), False, pygame.Color('LightGray'))
        t_residual = myfont.render('residual: ' + str(np.round(character.state[0:3]*yaml_p['unit'],4)), False, pygame.Color('LightGray'))
        if yaml_p['physics']:
            t_velocity = myfont.render('velocity: ' + str(np.round(character.state[3:6]*yaml_p['unit'],1)), False, pygame.Color('LightGray'))
            t_border = myfont.render('border: ' + str(np.round(character.state[6:12]*yaml_p['unit'],1)), False, pygame.Color('LightGray'))
            t_world_compressed = myfont.render('world_compressed: ' + str(np.round(character.state[12:],1)), False, pygame.Color('LightGray'))
        else:
            t_border = myfont.render('border: ' + str(np.round(character.state[3:6],1)), False, pygame.Color('LightGray'))
            t_world_compressed = myfont.render('world_compressed: ' + str(np.round(character.state[6:],1)), False, pygame.Color('LightGray'))

        screen.blit(t_reward_step,(30,10))
        screen.blit(t_reward_epi,(30,25))
        screen.blit(t_residual,(30,55))
        if yaml_p['physics']:
            screen.blit(t_velocity,(30,70))
        screen.blit(t_border,(30,85))
        screen.blit(t_world_compressed,(30,100))

    # updating the window
    pygame.display.flip()
    #clock.tick(10) #cycles per second
    clock.tick(10)
