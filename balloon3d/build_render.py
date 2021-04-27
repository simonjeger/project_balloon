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
    render_ratio = int(yaml_p['unit_xy'] / yaml_p['unit_z'])

    size_x = character.size_x*render_ratio
    size_y = character.size_y*render_ratio
    size_z = character.size_z

    # general setup
    pygame.init()
    clock = pygame.time.Clock()

    # setting up the main window
    res = 1.3 #int(100/size_z)
    screen_width = int(3*size_z*res)
    screen_height = int((size_y + 2*size_z)*res)
    screen = pygame.display.set_mode((screen_width, screen_height))

    c_background = (34,42,53)
    screen.fill(c_background)
    pygame.display.set_caption('balloon3d')

    # handling input
    for event in pygame.event.get():
        if event.type == pygame.QUIT: #does user click closing button
            pygame.quit()
            sys.exit()

    # read in wind_map
    world = torch.load('data/' + train_or_test + '/tensor/' + world_name + '.pt')

    # generate the three windows
    visualize_world(world, character.position)

    for dim in ['xz', 'yz', 'xy']:
        display_movement(dim, screen, int(screen_width), int(screen_height), size_x, size_y, size_z, render_ratio, window_size, res, character)

    # text
    myfont = pygame.font.SysFont('Arial', 10, bold=False)
    t_reward_step = myfont.render('reward_step: ' + str(np.round(reward_step,3)), False, pygame.Color('LightGray'))
    t_reward_epi = myfont.render('reward_epi: ' + str(np.round(reward_epi,3)), False, pygame.Color('LightGray'))
    t_residual = myfont.render('residual: ' + str(np.round(np.multiply(character.state[0:3],[yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']]),4)), False, pygame.Color('LightGray'))
    if yaml_p['physics']:
        t_velocity = myfont.render('velocity: ' + str(np.round(np.multiply(character.state[3:6],[yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']]),1)), False, pygame.Color('LightGray'))
        t_border_x = myfont.render('border_x: ' + str(np.round(np.multiply(character.state[6:8],[yaml_p['unit_xy'], yaml_p['unit_xy']]),1)), False, pygame.Color('LightGray'))
        t_border_y = myfont.render('border_y: ' + str(np.round(np.multiply(character.state[8:10],[yaml_p['unit_xy'], yaml_p['unit_xy']]),1)), False, pygame.Color('LightGray'))
        t_border_z = myfont.render('border_z: ' + str(np.round(np.multiply(character.state[10:12],[yaml_p['unit_z'], yaml_p['unit_z']]),1)), False, pygame.Color('LightGray'))
        t_world_compressed = myfont.render('world_compressed: ' + str(np.round(character.state[8:],1)), False, pygame.Color('LightGray'))
    else:
        t_border_x = myfont.render('border_x: ' + str(np.round(np.multiply(character.state[3:5],[yaml_p['unit_xy'], yaml_p['unit_xy']]),1)), False, pygame.Color('LightGray'))
        t_border_y = myfont.render('border_y: ' + str(np.round(np.multiply(character.state[5:7],[yaml_p['unit_xy'], yaml_p['unit_xy']]),1)), False, pygame.Color('LightGray'))
        t_border_z = myfont.render('border_z: ' + str(np.round(np.multiply(character.state[7:9],[yaml_p['unit_z'], yaml_p['unit_z']]),1)), False, pygame.Color('LightGray'))
        t_world_compressed = myfont.render('world_compressed: ' + str(np.round(character.state[9:],1)), False, pygame.Color('LightGray'))

    start_text = 2*size_z*res
    space_text = 15
    screen.blit(t_reward_step,(space_text,start_text+1*space_text))
    screen.blit(t_reward_epi,(space_text,start_text+2*space_text))
    screen.blit(t_residual,(space_text,start_text+3*space_text))
    if yaml_p['physics']:
        screen.blit(t_velocity,(space_text,start_text+4*space_text))
    screen.blit(t_border_x,(space_text,start_text+5*space_text))
    screen.blit(t_border_y,(space_text,start_text+6*space_text))
    screen.blit(t_border_z,(space_text,start_text+7*space_text))
    screen.blit(t_world_compressed,(space_text,start_text+8*space_text))

    # updating the window
    pygame.display.flip()
    clock.tick(10) #cycles per second

def display_movement(dim, screen, screen_width, screen_height, size_x, size_y, size_z, render_ratio, window_size, res, character):
    if dim == 'xz':
        i1 = 0
        i2 = 2
        size_1 = size_x
        size_2 = size_z
        dist_to_top = size_z
        dist_to_bottom = 2*size_z
        position_1 = character.position[i1]*render_ratio
        position_2 = character.position[i2]
        target_1 = character.target[i1]*render_ratio
        target_2 = character.target[i2]
        ceiling = character.ceiling[:,int(character.position[1])]

    if dim == 'yz':
        i1 = 1
        i2 = 2
        size_1 = size_y
        size_2 = size_z
        dist_to_top = 0
        dist_to_bottom = size_z
        position_1 = character.position[i1]*render_ratio
        position_2 = character.position[i2]
        target_1 = character.target[i1]*render_ratio
        target_2 = character.target[i2]
        ceiling = character.ceiling[int(character.position[0]),:]

    if dim == 'xy':
        i1 = 0
        i2 = 1
        size_1 = size_x
        size_2 = size_y
        position_1 = character.position[i1]*render_ratio
        position_2 = character.position[i2]*render_ratio
        target_1 = character.target[i1]*render_ratio
        target_2 = character.target[i2]*render_ratio
        dist_to_top = 2*size_z
        dist_to_bottom = size_y + 2*size_z

    window_size = window_size*render_ratio

    # colors
    c_ceiling = (51,63,80,150)
    c_down = (117,21,0,255)
    c_stay = (173,29,0)
    c_up = (237,35,1)
    c_path = (242,242,242)
    c_window = (217,217,217, 50)
    c_target_center = (242,242,242)
    c_target_radius = (217,217,217,50)

    left_border = (screen_width/2)/res
    right_border = size_1 - left_border

    if (position_1 < left_border): #left border
        offset = 0

        # write and display background
        bg = pygame.image.load('render/wind_map_' + dim + '.png')
        img_height = int(screen_height*size_2/(size_y + 2*size_z))
        img_width = int(img_height*size_1/size_2)
        bg = pygame.transform.scale(bg, (img_width, img_height))
        screen.blit(bg, (0, dist_to_top*res))

        if dim != 'xy':
            # write and display ceiling
            w_ceiling = []
            for i in range(len(ceiling)):
                pair = (((1+1/len(ceiling))*i*render_ratio*res,(dist_to_bottom - ceiling[i])*res))
                w_ceiling.append(pair)
            w_ceiling.append((size_1*res, dist_to_top))
            w_ceiling.append((0, dist_to_top))

            lx, ly = zip(*w_ceiling)
            min_x, min_y, max_x, max_y = min(lx), min(ly), max(lx), max(ly)
            target_rect = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)
            shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
            pygame.draw.polygon(shape_surf, c_ceiling, [(x - min_x, y - min_y) for x, y in w_ceiling])
            screen.blit(shape_surf, target_rect)

        # write and display observing box
        size_obs_x = window_size*2*res
        if dim != 'xy':
            size_obs_y = (dist_to_bottom - dist_to_top)*res
            pos_obs = [int(position_1 - window_size)*res, dist_to_top*res]
        else:
            size_obs_y = window_size*2*res
            pos_obs = [int(position_1 - window_size)*res, int(dist_to_bottom - position_2 - window_size)*res]
        rec_obs = pygame.Rect(pos_obs[0], pos_obs[1], size_obs_x, size_obs_y)

        shape_surf = pygame.Surface(pygame.Rect(rec_obs).size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, c_window, shape_surf.get_rect())
        screen.blit(shape_surf, rec_obs)

        # write path
        path = []
        if dim != 'xy':
            for i in character.path:
                path.append((i[i1]*render_ratio*res, (dist_to_bottom-i[i2])*res))
        else:
            for i in character.path:
                path.append((i[i1]*render_ratio*res, (dist_to_bottom-i[i2]*render_ratio)*res))

        # write balloon
        size_balloon = 4*res
        pos_balloon = [position_1*res, (dist_to_bottom - position_2)*res]
        rec_balloon = pygame.Rect(pos_balloon[0] - size_balloon/2, pos_balloon[1] - size_balloon/2, size_balloon, size_balloon)

        # write target
        size_target = 4*res
        pos_target = [target_1*res, (dist_to_bottom - target_2)*res]
        rec_target = pygame.Rect(pos_target[0] - size_target/2, pos_target[1] - size_target/2, size_target, size_target)

    elif (position_1 >= right_border): #right border
        offset = -right_border + screen_width/2/res

        # write and display background
        bg = pygame.image.load('render/wind_map_' + dim + '.png')
        img_height = int(screen_height*size_2/(size_y + 2*size_z))
        img_width = int(img_height*size_1/size_2)
        bg = pygame.transform.scale(bg, (img_width, img_height))
        screen.blit(bg, (int(offset*res), dist_to_top*res))

        if dim != 'xy':
            # write and display ceiling
            w_ceiling = []
            for i in range(len(ceiling)):
                pair = (((1+1/len(ceiling))*(i*render_ratio+offset)*res,(dist_to_bottom - ceiling[i])*res))
                w_ceiling.append(pair)
            w_ceiling.append((size_1*res, dist_to_top))
            w_ceiling.append((0, dist_to_top))

            lx, ly = zip(*w_ceiling)
            min_x, min_y, max_x, max_y = min(lx), min(ly), max(lx), max(ly)
            target_rect = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)
            shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
            pygame.draw.polygon(shape_surf, c_ceiling, [(x - min_x, y - min_y) for x, y in w_ceiling])
            screen.blit(shape_surf, target_rect)

        # write and display observing box
        size_obs_x = window_size*2*res
        if dim != 'xy':
            size_obs_y = (dist_to_bottom - dist_to_top)*res
            pos_obs = [int(position_1 + offset - window_size)*res, dist_to_top*res]
        else:
            size_obs_y = window_size*2*res
            pos_obs = [int(position_1 + offset - window_size)*res, int(dist_to_bottom - position_2 - window_size)*res]
        rec_obs = pygame.Rect(pos_obs[0], pos_obs[1], size_obs_x, size_obs_y)

        shape_surf = pygame.Surface(pygame.Rect(rec_obs).size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, c_window, shape_surf.get_rect())
        screen.blit(shape_surf, rec_obs)

        # write path
        path = []
        if dim != 'xy':
            for i in character.path:
                path.append(((i[i1]*render_ratio+offset)*res, (dist_to_bottom-i[i2])*res))
        else:
            for i in character.path:
                path.append(((i[i1]*render_ratio+offset)*res, (dist_to_bottom*render_ratio-i[i2])*res))

        # write balloon
        size_balloon = 4*res
        pos_balloon = [(position_1 + offset)*res, (dist_to_bottom - position_2)*res]
        rec_balloon = pygame.Rect(pos_balloon[0] - size_balloon/2, pos_balloon[1] - size_balloon/2, size_balloon, size_balloon)

        # write target
        size_target = 4*res
        pos_target = [(target_1 + offset)*res, (dist_to_bottom - target_2)*res]
        rec_target = pygame.Rect(pos_target[0] - size_target/2, pos_target[1] - size_target/2, size_target, size_target)

    else: #in between
        offset = left_border - position_1

        # write and display background
        bg = pygame.image.load('render/wind_map_' + dim + '.png')
        img_height = int(screen_height*size_2/(size_y + 2*size_z))
        img_width = int(img_height*size_1/size_2)
        bg = pygame.transform.scale(bg, (img_width, img_height))
        screen.blit(bg, (int(offset*res), dist_to_top*res))

        if dim != 'xy':
            # draw and display ceiling
            w_ceiling = []
            for i in range(len(ceiling)):
                pair = (((1+1/len(ceiling))*(i*render_ratio+offset)*res,(dist_to_bottom - ceiling[i])*res))
                w_ceiling.append(pair)
            w_ceiling.append((size_1*res, dist_to_top))
            w_ceiling.append((0, dist_to_top))

            lx, ly = zip(*w_ceiling)
            min_x, min_y, max_x, max_y = min(lx), min(ly), max(lx), max(ly)
            target_rect = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)
            shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
            pygame.draw.polygon(shape_surf, c_ceiling, [(x - min_x, y - min_y) for x, y in w_ceiling])
            screen.blit(shape_surf, target_rect)

        # write and display observing box
        size_obs_x = window_size*2*res
        if dim != 'xy':
            size_obs_y = (dist_to_bottom - dist_to_top)*res
            pos_obs = [int(position_1 + offset - window_size)*res, dist_to_top*res]
        else:
            size_obs_y = window_size*2*res
            pos_obs = [int(position_1 + offset - window_size)*res, int(dist_to_bottom - position_2 - window_size)*res]
        rec_obs = pygame.Rect(pos_obs[0], pos_obs[1], size_obs_x, size_obs_y)

        shape_surf = pygame.Surface(pygame.Rect(rec_obs).size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, c_window, shape_surf.get_rect())
        screen.blit(shape_surf, rec_obs)

        # write path
        path = []
        if dim != 'xy':
            for i in character.path:
                path.append(((i[i1]*render_ratio+offset)*res, (dist_to_bottom-i[i2])*res))
        else:
            for i in character.path:
                path.append(((i[i1]*render_ratio+offset)*res, (dist_to_bottom-i[i2]*render_ratio)*res))

        # write balloon
        size_balloon = 4*res
        pos_balloon = [screen_width/2, (dist_to_bottom - position_2)*res]
        rec_balloon = pygame.Rect(pos_balloon[0] - size_balloon/2, pos_balloon[1] - size_balloon/2, size_balloon, size_balloon)

        # write target
        size_target = 4*res
        pos_target = [(target_1 + offset)*res, (dist_to_bottom - target_2)*res]
        rec_target = pygame.Rect(pos_target[0] - size_target/2, pos_target[1] - size_target/2, size_target, size_target)

    # balloon
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

    # draw minimap
    length_mm = screen_width/10
    height_mm = length_mm/size_1*size_2
    position_mm = [screen_width - 20 - length_mm, dist_to_top*res + 20]
    rec_obs = pygame.Rect(position_mm[0], position_mm[1], length_mm, height_mm)
    shape_surf = pygame.Surface(pygame.Rect(rec_obs).size, pygame.SRCALPHA)
    pygame.draw.rect(shape_surf, c_window, shape_surf.get_rect())
    screen.blit(shape_surf, rec_obs)

    mm_balloon = [position_1/size_1*length_mm, height_mm - position_2/size_2*height_mm]
    position_balloon_mm = [position_mm[0]+mm_balloon[0], position_mm[1]+mm_balloon[1]]
    pygame.draw.circle(screen, c_stay, position_balloon_mm, 1)


    mm_target = [target_1/size_1*length_mm, height_mm - target_2/size_2*height_mm]
    position_target_mm = [position_mm[0]+mm_target[0], position_mm[1]+mm_target[1]]
    pygame.draw.circle(screen, c_target_radius, position_target_mm, 1)
