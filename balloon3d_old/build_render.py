from visualize_wind_map import visualize_wind_map

import numpy as np
import pygame, sys

def build_render(character, reward_step, reward_epi, window_size, train_or_test):
    size_x = character.size_x
    size_z = character.size_z

    # general setup
    pygame.init()
    clock = pygame.time.Clock()

    # setting up the main window
    res = 40
    wall = 1
    screen_width = 2*size_z * res
    screen_height = size_z * res
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('balloon3d')

    # handling input
    for event in pygame.event.get():
        if event.type == pygame.QUIT: #does user click closing button
            pygame.quit()
            sys.exit()

    display_movement('yz', [0,0], screen, int(screen_width), int(screen_height), size_x, size_z, window_size, res, character)

    # text
    myfont = pygame.font.SysFont('Arial', 15, bold = True)
    time_remaining = myfont.render('time remaining: ' + str(character.t), False, pygame.Color('LightGray'))
    reward_step = myfont.render('reward_step: ' + str(round(reward_step,2)), False, pygame.Color('LightGray'))
    reward_epi = myfont.render('reward_epi: ' + str(round(reward_epi,2)), False, pygame.Color('LightGray'))
    residual = myfont.render('residual: ' + str(np.round(character.state[0:2],2)), False, pygame.Color('LightGray'))
    border = myfont.render('border: ' + str(np.round(character.state[2:6],2)), False, pygame.Color('LightGray'))
    wind_compressed = myfont.render('wind_compressed: ' + str(np.round(character.state[7:],2)), False, pygame.Color('LightGray'))

    screen.blit(time_remaining,(10,10))
    screen.blit(reward_step,(10,25))
    screen.blit(reward_epi,(10,40))
    screen.blit(residual,(10,55))
    screen.blit(border,(10,70))
    screen.blit(wind_compressed,(10,85))

    # updating the window
    pygame.display.flip()
    clock.tick(10) #cycles per second


def display_movement(plane, orgin, screen, screen_width, screen_height, size_1, size_2, window_size, res, character):
    if plane == 'xz':
        i1 = 0
        i2 = 2
    if plane == 'yz':
        i1 = 1
        i2 = 2
    if plane == 'xy':
        i1 = 0
        i2 = 1

    left_border = (screen_width/2)/res
    right_border = size_1 - left_border

    if (character.position[i1] < left_border): #left border
        offset = 0
        # game rectangles (orgin is always on the top left, so increase y to go down)
        size_obs_x = window_size*2*res
        size_obs_y = size_2*2*res
        pos_obs = [character.position[i1]*res, size_2/2*res]
        rec_obs = pygame.Rect(pos_obs[0] - size_obs_x/2, pos_obs[1] - size_obs_y/2, size_obs_x, size_obs_y)

        path = []
        for i in character.path:
            path.append((i[i1]*res, (size_2-i[i2])*res))

        size_balloon = 0.5*res
        pos_balloon = [character.position[i1]*res, (size_2 - character.position[i2])*res]
        rec_balloon = pygame.Rect(pos_balloon[0] - size_balloon/2, pos_balloon[1] - size_balloon/2, size_balloon, size_balloon)

        size_target = 0.5*res
        pos_target = [character.target[i1]*res, (size_2 - character.target[i2])*res]
        rec_target = pygame.Rect(pos_target[0] - size_target/2, pos_target[1] - size_target/2, size_target, size_target)

        # generate background
        screen.fill(pygame.Color('grey12'))
        pygame.draw.rect(screen, pygame.Color('DimGrey'), rec_obs) #draw window
        visualize_wind_map(character.wind_map, character.position) #generate background image
        bg = pygame.image.load('render/wind_map_xz.png')
        size = bg.get_rect().size
        bg = pygame.transform.scale(bg, (int(screen_height*size[0]/size[1]), screen_height))
        screen.blit(bg, (0, 0))

    elif (character.position[0] >= right_border): #right border
        offset = -right_border/2
        # game rectangles (orgin is always on the top left, so increase y to go down)
        size_obs_x = window_size*2*res
        size_obs_y = size_2*2*res
        pos_obs = [(character.position[i1] + offset)*res, size_2/2*res]
        rec_obs = pygame.Rect(pos_obs[0] - size_obs_x/2, pos_obs[1] - size_obs_y/2, size_obs_x, size_obs_y)

        path = []
        for i in character.path:
            path.append(((i[i1]+offset)*res, (size_2-i[i2])*res))

        size_balloon = 0.5*res
        pos_balloon = [(character.position[i1] + offset)*res, (size_2 - character.position[i2])*res]
        rec_balloon = pygame.Rect(pos_balloon[0] - size_balloon/2, pos_balloon[1] - size_balloon/2, size_balloon, size_balloon)

        size_target = 0.5*res
        pos_target = [(character.target[i1] + offset)*res, (size_2 - character.target[i2])*res]
        rec_target = pygame.Rect(pos_target[0] - size_target/2, pos_target[1] - size_target/2, size_target, size_target)

        # generate background
        screen.fill(pygame.Color('grey12'))
        pygame.draw.rect(screen, pygame.Color('DimGrey'), rec_obs) #draw window
        visualize_wind_map(character.wind_map, character.position) #generate background image
        bg = pygame.image.load('render/wind_map_xz.png')
        size = bg.get_rect().size
        bg = pygame.transform.scale(bg, (int(screen_height*size[0]/size[1]), screen_height))
        screen.blit(bg, (int(offset*res), 0))

    else: #in between
        offset = left_border - character.position[i1]
        # game rectangles (orgin is always on the top left, so increase y to go down)
        size_obs_x = window_size*2*res
        size_obs_y = size_2*2*res
        pos_obs = [screen_width/2, size_2/2*res]
        rec_obs = pygame.Rect(pos_obs[0] - size_obs_x/2, pos_obs[1] - size_obs_y/2, size_obs_x, size_obs_y)

        path = []
        for i in character.path:
            path.append(((i[i1]+offset)*res, (size_2-i[i2])*res))

        size_balloon = 0.5*res
        pos_balloon = [screen_width/2, (size_2 - character.position[i2])*res]
        rec_balloon = pygame.Rect(pos_balloon[0] - size_balloon/2, pos_balloon[1] - size_balloon/2, size_balloon, size_balloon)

        size_target = 0.5*res
        pos_target = [(character.target[i1] + offset)*res, (size_2 - character.target[i2])*res]
        rec_target = pygame.Rect(pos_target[0] - size_target/2, pos_target[1] - size_target/2, size_target, size_target)

        # generate background
        screen.fill(pygame.Color('grey12'))
        pygame.draw.rect(screen, pygame.Color('DimGrey'), rec_obs) #draw window
        visualize_wind_map(character.wind_map, character.position) #generate background image
        bg = pygame.image.load('render/wind_map_xz.png')
        size = bg.get_rect().size
        bg = pygame.transform.scale(bg, (int(screen_height*size[0]/size[1]), screen_height))
        screen.blit(bg, (int(offset*res), 0))

    # visuals
    if len(character.path) > 1:
        pygame.draw.lines(screen, pygame.Color('LightGray'), False, path, 1)
    if character.action == 2:
        pygame.draw.ellipse(screen, pygame.Color('PaleTurquoise'), rec_balloon)
    if character.action == 1:
        pygame.draw.ellipse(screen, pygame.Color('LightGray'), rec_balloon)
    if character.action == 0:
        pygame.draw.ellipse(screen, pygame.Color('Plum'), rec_balloon)
    pygame.draw.ellipse(screen, pygame.Color('FireBrick'), rec_target)
