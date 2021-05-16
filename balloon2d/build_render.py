import numpy as np
import pygame, sys
import matplotlib.pylab as pl

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

def build_render(character, reward_step, reward_epi, world_name, window_size, radius_z, train_or_test, roll_out):
    render_ratio = int(yaml_p['unit_xy'] / yaml_p['unit_z'])

    size_x = character.size_x
    size_z = character.size_z

    # general setup
    pygame.init()
    clock = pygame.time.Clock()

    # setting up the main window
    res = 480/size_z
    screen_width = int(3*size_z*res)
    screen_height = int(size_z*res)
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('balloon2d')

    # handling input
    for event in pygame.event.get():
        if event.type == pygame.QUIT: #does user click closing button
            pygame.quit()
            sys.exit()

    display_movement(screen, int(screen_width), int(screen_height), size_x, size_z, render_ratio, window_size, radius_z, res, character, world_name, train_or_test, roll_out)

    # text
    myfont = pygame.font.SysFont('Arial', 10, bold = False)
    t_reward_step = myfont.render('reward_step: ' + str(np.round(reward_step,3)), False, pygame.Color('LightGray'))
    t_reward_epi = myfont.render('reward_epi: ' + str(np.round(reward_epi,3)), False, pygame.Color('LightGray'))
    t_residual = myfont.render('residual: ' + str(np.round(np.multiply(character.state[0:2],[yaml_p['unit_xy'], yaml_p['unit_z']]),4)), False, pygame.Color('LightGray'))
    if yaml_p['physics']:
        t_velocity = myfont.render('velocity: ' + str(np.round(np.multiply(character.state[2:4],[yaml_p['unit_xy'], yaml_p['unit_z']]),1)), False, pygame.Color('LightGray'))
        t_measurement = myfont.render('measurement: ' + str(np.round(np.multiply(character.state[4:6],[yaml_p['unit_xy'], yaml_p['unit_z']]),1)), False, pygame.Color('LightGray'))
        t_border_x = myfont.render('border_x: ' + str(np.round(character.state[6]*yaml_p['unit_xy'],1)), False, pygame.Color('LightGray'))
        t_border_z = myfont.render('border_z: ' + str(np.round(character.state[7]*yaml_p['unit_z'],1)), False, pygame.Color('LightGray'))
        t_world_compressed = myfont.render('world_compressed: ' + str(np.round(character.state[8:],1)), False, pygame.Color('LightGray'))
    else:
        t_measurement = myfont.render('measurement: ' + str(np.round(np.multiply(character.state[2:4],[yaml_p['unit_xy'], yaml_p['unit_z']]),1)), False, pygame.Color('LightGray'))
        t_border_x = myfont.render('border_x: ' + str(np.round(character.state[4]*yaml_p['unit_xy'],1)), False, pygame.Color('LightGray'))
        t_border_z = myfont.render('border_z: ' + str(np.round(character.state[5]*yaml_p['unit_z'],1)), False, pygame.Color('LightGray'))
        t_world_compressed = myfont.render('world_compressed: ' + str(np.round(character.state[6:],1)), False, pygame.Color('LightGray'))


    screen.blit(t_reward_step,(50,10))
    screen.blit(t_reward_epi,(50,25))
    screen.blit(t_residual,(50,55))
    if yaml_p['physics']:
        screen.blit(t_velocity,(50,70))
    screen.blit(t_measurement,(50,85))
    screen.blit(t_border_x,(50,100))
    screen.blit(t_border_z,(50,115))
    screen.blit(t_world_compressed,(50,130))

    # updating the window
    pygame.display.flip()
    clock.tick(10) #cycles per second


def display_movement(screen, screen_width, screen_height, size_x, size_z, render_ratio, window_size, radius_z, res, character, world_name, train_or_test, roll_out):
    # for this 2d case I only look at xz
    i1 = 0
    i2 = 1

    size_1 = size_x*render_ratio
    size_2 = size_z

    position_1 = character.position[i1]*render_ratio
    position_2 = character.position[i2]

    target_1 = character.target[i1]*render_ratio
    target_2 = character.target[i2]

    window_size = window_size*render_ratio

    # colors
    c_background = (34,42,53)
    c_ceiling = (150,150,150,150)
    c_down = (117,21,0,255)
    c_stay = (173,29,0)
    c_up = (237,35,1)
    c_path = (242,242,242)
    c_path_roll_out = (110,110,110)
    c_window = (217,217,217, 50)
    c_target_center = (242,242,242)
    c_target_radius = (217,217,217,50)
    c_continuous = 'seismic'

    left_border = (screen_width/2)/res
    right_border = size_1 - left_border

    if (position_1 < left_border): #left border
        offset = 0
    elif (position_1 >= right_border): #right border
        offset = -right_border + screen_width/2/res
    else: #in between
        offset = left_border - position_1

    # write and display background
    screen.fill(c_background)
    bg = pygame.image.load(yaml_p['data_path'] + train_or_test + '/image/' + world_name + '.png')
    size = bg.get_rect().size
    bg = pygame.transform.scale(bg, (int(screen_height*size[0]/size[1]), screen_height))
    screen.blit(bg, (int(offset*res), 0))

    # draw and display ceiling
    ceiling = []
    for i in range(len(character.ceiling)):
        pair = (((1+1/len(character.ceiling))*(i*render_ratio+offset)*res,(size_2 - character.ceiling[i])*res))
        ceiling.append(pair)
    ceiling.append((size_1*res, 0))
    ceiling.append((0, 0))

    lx, ly = zip(*ceiling)
    min_x, min_y, max_x, max_y = min(lx), min(ly), max(lx), max(ly)
    target_rect = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)
    shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
    pygame.draw.polygon(shape_surf, c_ceiling, [(x - min_x, y - min_y) for x, y in ceiling])
    screen.blit(shape_surf, target_rect)

    # write and display observing box
    size_obs_x = (window_size*2 + 1*render_ratio)*res
    size_obs_y = size_2*res
    pos_obs = [(int(position_1/render_ratio)*render_ratio + offset - window_size)*res, 0]
    rec_obs = pygame.Rect(pos_obs[0], pos_obs[1], size_obs_x, size_obs_y)

    shape_surf = pygame.Surface(pygame.Rect(rec_obs).size, pygame.SRCALPHA)
    pygame.draw.rect(shape_surf, c_window, shape_surf.get_rect())
    screen.blit(shape_surf, rec_obs)

    # write path
    path = []
    for i in character.path:
        path.append(((i[i1]*render_ratio+offset)*res, (size_2-i[i2])*res))

    if roll_out is not None:
        path_roll_out = []
        for i in roll_out:
            path_roll_out.append(((i[i1]*render_ratio+offset)*res, (size_2-i[i2])*res))

    # write balloon
    size_balloon = 2*res
    pos_balloon = [(position_1 + offset)*res, (size_2 - position_2)*res]
    rec_balloon = pygame.Rect(pos_balloon[0] - size_balloon/2, pos_balloon[1] - size_balloon/2, size_balloon, size_balloon)

    # write target
    size_target = 2*res
    pos_target = [(target_1 + offset)*res, (size_2 - target_2)*res]
    rec_target = pygame.Rect(pos_target[0] - size_target/2, pos_target[1] - size_target/2, size_target, size_target)

    # path
    if roll_out is not None:
        pygame.draw.lines(screen, c_path_roll_out, False, path_roll_out, 1)

    if len(character.path) > 1:
        pygame.draw.lines(screen, c_path, False, path, 1)

    # balloon
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

    # draw target (dense and transparent)
    pygame.draw.ellipse(screen, c_target_center, rec_target)

    r_x = yaml_p['radius_x']*res
    r_z = radius_z*res

    target_rect = pygame.Rect((pos_target), (0, 0)).inflate((r_x * 2, r_z * 2))
    shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
    pygame.draw.ellipse(shape_surf, c_target_radius, (0, 0, 2*r_x, 2*r_z))
    screen.blit(shape_surf, target_rect)

    # draw position bar
    length = screen_width/10
    point = position_1/size_1*length
    position_bar = [[screen_width - 20 - length, 20], [screen_width - 20, 20]]
    pygame.draw.lines(screen, c_path, False, position_bar, 1)
    position_point = [position_bar[0][0]+point, position_bar[0][1]]
    pygame.draw.circle(screen, c_target_radius, position_point, 2)
