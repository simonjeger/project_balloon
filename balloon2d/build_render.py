import numpy as np
import torch
import pygame, sys
import matplotlib.pylab as pl

from visualize_world import visualize_world

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

def build_render(character, reward_step, reward_epi, world_name, window_size, radius_x, radius_z, train_or_test, roll_out):
    render_ratio = int(yaml_p['unit_x'] / yaml_p['unit_z'])

    size_x = character.size_x*render_ratio
    size_z = character.size_z

    # general setup
    pygame.init()
    clock = pygame.time.Clock()

    # setting up the main window
    res = 4 #int(100/size_z)
    screen_width = int(3*size_z*res)
    screen_height = int(size_z*res)
    screen = pygame.display.set_mode((screen_width, screen_height))

    c_background = (34,42,53)
    screen.fill(c_background)
    pygame.display.set_caption('balloon2d')

    # handling input
    for event in pygame.event.get():
        if event.type == pygame.QUIT: #does user click closing button
            pygame.quit()
            sys.exit()

    # read in wind_map
    world = torch.load(yaml_p['data_path'] + train_or_test + '/tensor/' + world_name + '.pt')

    # generate the three windows
    visualize_world(world, character.position)

    for dim in ['xz']:
        display_movement(dim, screen, screen_width, screen_height, c_background, size_x, size_z, render_ratio, window_size, radius_x, radius_z, res, character, roll_out)

    # text
    myfont = pygame.font.SysFont('Arial', 10, bold=False)

    t_reward_step = myfont.render('reward_step: ' + '{:.3f}'.format(reward_step), False, pygame.Color('LightGray'))
    t_reward_epi = myfont.render('reward_epi: ' + '{:.3f}'.format(reward_epi), False, pygame.Color('LightGray'))
    t_residual = myfont.render('residual: ' + str([round(num, 3) for num in character.state[0:2].tolist()]), False, pygame.Color('LightGray'))
    t_velocity = myfont.render('velocity: ' + str([round(num, 3) for num in character.state[2:4].tolist()]), False, pygame.Color('LightGray'))

    if yaml_p['type'] == 'regular':
        if yaml_p['boundaries'] == 'short':
            t_border_x = myfont.render('border_x: ' + '{:.3f}'.format(character.state[4]), False, pygame.Color('LightGray'))
            t_border_z = myfont.render('border_z: ' + '{:.3f}'.format(character.state[5]), False, pygame.Color('LightGray'))
            t_rel_pos = myfont.render('rel_pos: ' + '{:.3f}'.format(character.state[6]), False, pygame.Color('LightGray'))
            t_measurement = myfont.render('measurement: ' + str([round(num, 3) for num in character.state[7:9].tolist()]), False, pygame.Color('LightGray'))
            t_world_compressed = myfont.render('world_compressed: ' + str([round(num, 3) for num in character.state[8:].tolist()]), False, pygame.Color('LightGray'))
        if yaml_p['boundaries'] == 'long':
            t_border_x = myfont.render('border_x: ' + str([round(num, 3) for num in character.state[4:6].tolist()]), False, pygame.Color('LightGray'))
            t_border_z = myfont.render('border_z: ' + str([round(num, 3) for num in character.state[6:9].tolist()]), False, pygame.Color('LightGray'))
            t_measurement = myfont.render('measurement: ' + str([round(num, 3) for num in character.state[9:11].tolist()]), False, pygame.Color('LightGray'))
            t_world_compressed = myfont.render('world_compressed: ' + str([round(num, 3) for num in character.state[11:].tolist()]), False, pygame.Color('LightGray'))

    elif yaml_p['type'] == 'squished':
        if yaml_p['boundaries'] == 'short':
            t_border_x = myfont.render(' ', False, pygame.Color('LightGray'))
            t_border_z = myfont.render(' ', False, pygame.Color('LightGray'))
            t_rel_pos = myfont.render('rel_pos: ' + '{:.3f}'.format(character.state[4]), False, pygame.Color('LightGray'))
            t_measurement = myfont.render('measurement: ' + str([round(num, 3) for num in character.state[5:7].tolist()]), False, pygame.Color('LightGray'))
            t_world_compressed = myfont.render('world_compressed: ' + str([round(num, 3) for num in character.state[7:].tolist()]), False, pygame.Color('LightGray'))
        if yaml_p['boundaries'] == 'long':
            t_border_x = myfont.render('border_x: ' + str([round(num, 3) for num in character.state[4:6].tolist()]), False, pygame.Color('LightGray'))
            t_border_z = myfont.render('border_z: ' + str([round(num, 3) for num in character.state[6:8].tolist()]), False, pygame.Color('LightGray'))
            t_measurement = myfont.render('measurement: ' + str([round(num, 3) for num in character.state[8:10].tolist()]), False, pygame.Color('LightGray'))
            t_world_compressed = myfont.render('world_compressed: ' + str([round(num, 3) for num in character.state[10:].tolist()]), False, pygame.Color('LightGray'))

    screen.blit(t_reward_step,(10,10))
    screen.blit(t_reward_epi,(10,25))
    screen.blit(t_residual,(10,55))
    screen.blit(t_velocity,(10,70))
    screen.blit(t_border_x,(10,85))
    screen.blit(t_border_z,(10,100))
    if yaml_p['boundaries'] == 'short':
        screen.blit(t_rel_pos,(10,115))
    screen.blit(t_measurement,(10,130))
    screen.blit(t_world_compressed,(10,160))

    # updating the window
    pygame.display.flip()
    clock.tick(5) #cycles per second

def display_movement(dim, screen, screen_width, screen_height, c_background, size_x, size_z, render_ratio, window_size, radius_x, radius_z, res, character, roll_out):
    size_y = 0

    if dim == 'xz':
        i1 = 0
        i2 = 1
        size_1 = size_x
        size_2 = size_z
        dist_to_top = 0
        dist_to_bottom = size_z
        position_1 = character.position[i1]*render_ratio
        position_2 = character.position[i2]
        target_1 = character.target[i1]*render_ratio
        target_2 = character.target[i2]
        ceiling = character.ceiling

    window_size = window_size*render_ratio

    # colors
    c_ceiling = (150,150,150,150)
    c_down = (117,21,0,255)
    c_stay = (173,29,0)
    c_up = (237,35,1)
    c_path = (242,242,242)
    c_path_roll_out = (130,130,130)
    c_window = (217,217,217, 50)
    c_target_center = (242,242,242)
    c_target_radius = (217,217,217,50)

    left_border = (screen_width/2)/res
    right_border = size_1 - left_border

    if (position_1 < left_border): #left border
        offset_1 = 0
    elif (position_1 >= right_border): #right border
        offset_1 = -right_border + left_border
    else: #in between
        offset_1 = left_border - position_1

    lower_border = (screen_width/2)/res
    upper_border = size_2 - lower_border
    if (position_2 < lower_border): #lower border
        offset_2 = 0
    elif (position_2 >= upper_border): #upper border
        offset_2 = -upper_border + lower_border
    else: #in between
        offset_2 = lower_border - position_2

    # write and display background
    bg = pygame.image.load('render/wind_map_' + dim + '.png')
    img_height = int(size_2*res)
    img_width = int(img_height*size_1/size_2)
    bg = pygame.transform.scale(bg, (img_width, img_height))

    if dim != 'xy':
        screen.blit(bg, (offset_1*res, dist_to_top*res))
    else:
        screen.blit(bg, (offset_1*res, (dist_to_top-offset_2 - upper_border + lower_border)*res))

    if dim != 'xy':
        # draw and display ceiling
        size_ceil_x = size_1*res
        size_ceil_y = (size_2-ceiling)*res
        pos_ceil = [0, dist_to_top*res]
        rec_ceil = pygame.Rect(pos_ceil[0], pos_ceil[1], size_ceil_x, size_ceil_y)
        shape_surf = pygame.Surface(pygame.Rect(rec_ceil).size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, c_ceiling, shape_surf.get_rect())
        screen.blit(shape_surf, rec_ceil)

    # write and display observing box
    size_obs_x = (window_size*2+1*render_ratio)*res
    if dim != 'xy':
        size_obs_y = (dist_to_bottom - dist_to_top)*res
        pos_obs = [(int(position_1/render_ratio)*render_ratio + offset_1 - window_size)*res, dist_to_top*res]
    else:
        size_obs_y = (window_size*2+1*render_ratio)*res
        pos_obs = [(int(position_1/render_ratio)*render_ratio + offset_1 - window_size)*res, (int((dist_to_bottom - position_2)/render_ratio)*render_ratio - offset_2 - window_size)*res]
    rec_obs = pygame.Rect(pos_obs[0], pos_obs[1], size_obs_x, size_obs_y)

    shape_surf = pygame.Surface(pygame.Rect(rec_obs).size, pygame.SRCALPHA)
    pygame.draw.rect(shape_surf, c_window, shape_surf.get_rect())
    screen.blit(shape_surf, rec_obs)

    # write path
    path = []
    if dim != 'xy':
        for i in character.path:
            path.append(((i[i1]*render_ratio+offset_1)*res, (dist_to_bottom-i[i2])*res))
    else:
        for i in character.path:
            path.append(((i[i1]*render_ratio+offset_1)*res, (dist_to_bottom-i[i2]*render_ratio-offset_2)*res))

    if roll_out is not None:
        path_roll_out = []
        if dim != 'xy':
            for i in roll_out:
                path_roll_out.append(((i[i1]*render_ratio+offset_1)*res, (dist_to_bottom-i[i2])*res))
        else:
            for i in roll_out:
                path_roll_out.append(((i[i1]*render_ratio+offset_1)*res, (dist_to_bottom-i[i2]*render_ratio-offset_2)*res))

    # write balloon
    size_balloon = 2*res
    if dim != 'xy':
        pos_balloon = [(position_1 + offset_1)*res, (dist_to_bottom - position_2)*res]
    else:
        pos_balloon = [(position_1 + offset_1)*res, (dist_to_bottom - position_2 - offset_2)*res]
    rec_balloon = pygame.Rect(pos_balloon[0] - size_balloon/2, pos_balloon[1] - size_balloon/2, size_balloon, size_balloon)

    # write target
    size_target = 2*res
    if dim != 'xy':
        pos_target = [(target_1 + offset_1)*res, (dist_to_bottom - target_2)*res]
    else:
        pos_target = [(target_1 + offset_1)*res, (dist_to_bottom - target_2 - offset_2)*res]
    rec_target = pygame.Rect(pos_target[0] - size_target/2, pos_target[1] - size_target/2, size_target, size_target)

    # path
    if len(path_roll_out) > 1:
        pygame.draw.lines(screen, c_path_roll_out, False, path_roll_out, 1)

    if len(path) > 1:
        pygame.draw.lines(screen, c_path, False, path, 1)

    # balloon
    if yaml_p['type'] == 'regular':
        if yaml_p['continuous']:
            cv = 100
            colors = pl.cm.BrBG(np.linspace(0,1,cv+1))
            action = np.sign(character.action-1)*(abs(character.action - 1)**0.5 + 1)/2 #rescale the color map to make change more visible
            color = colors[int(action*cv)]*255
            pygame.draw.ellipse(screen, color, rec_balloon)
        else:
            if character.action == 2:
                pygame.draw.ellipse(screen, c_up, rec_balloon)
            if character.action == 1:
                pygame.draw.ellipse(screen, c_stay, rec_balloon)
            if character.action == 0:
                pygame.draw.ellipse(screen, c_down, rec_balloon)
    elif yaml_p['type'] == 'squished':
        cv = 100
        colors = pl.cm.BrBG(np.linspace(0,1,cv+1))
        action = character.action
        color = colors[int(action*cv)]*255
        pygame.draw.ellipse(screen, color, rec_balloon)

    # draw target (dense and transparent)
    pygame.draw.ellipse(screen, c_target_center, rec_target)

    if dim != 'xy':
        r_1 = radius_x*res
        r_2 = radius_z*res
    else:
        r_1 = radius_x*res
        r_2 = radius_x*res

    target_rect = pygame.Rect((pos_target), (0, 0)).inflate((r_1*2, r_2*2))
    shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
    pygame.draw.ellipse(shape_surf, c_target_radius, (0, 0, 2*r_1, 2*r_2))
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

    # overlay everything that overlaps from the xy plane
    if dim == 'xy':
        pygame.draw.rect(screen, c_background, pygame.Rect(0, 0, screen_width, dist_to_top*res))
