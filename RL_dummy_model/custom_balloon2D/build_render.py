import numpy as np
import pygame, sys

def build_render(character, reward_step, reward_epi, world_name, window_size, train_or_test):
    size_x = character.size_x
    size_z = character.size_z

    # general setup
    pygame.init()
    clock = pygame.time.Clock()

    # setting up the main window
    res = 40
    screen_width = size_x * res
    screen_height = size_z * res
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('balloon2d')

    # game rectangles (orgin is always on the top left, so increase y to go down)
    size_obs_x = window_size*2*res
    size_obs_y = size_z*2*res
    pos_obs = [character.position[0]*res, size_z/2*res]
    rec_obs = pygame.Rect(pos_obs[0] - size_obs_x/2, pos_obs[1] - size_obs_y/2, size_obs_x, size_obs_y)

    path = []
    for i in character.path:
        path.append((i[0]*res, (size_z-i[1])*res))

    size_balloon = 0.5*res
    pos_balloon = [character.position[0]*res, (size_z - character.position[1])*res]
    rec_balloon = pygame.Rect(pos_balloon[0] - size_balloon/2, pos_balloon[1] - size_balloon/2, size_balloon, size_balloon)

    size_target = 0.5*res
    pos_target = [character.target[0]*res, (size_z - character.target[1])*res]
    rec_target = pygame.Rect(pos_target[0] - size_target/2, pos_target[1] - size_target/2, size_target, size_target)

    # handling input
    for event in pygame.event.get():
        if event.type == pygame.QUIT: #does user click closing button
            pygame.quit()
            sys.exit()

    # generate background
    screen.fill(pygame.Color('grey12'))
    pygame.draw.rect(screen, pygame.Color('DimGrey'), rec_obs) #draw window
    bg = pygame.image.load('data/' + train_or_test + '/image/' + world_name + '.png')
    bg = pygame.transform.scale(bg, (screen_width, screen_height))
    screen.blit(bg, (0, 0))

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

    # text
    myfont = pygame.font.SysFont('Arial', 15, bold = True)
    reward_step = myfont.render('reward_step: ' + str(round(reward_step,4)), False, pygame.Color('LightGray'))
    reward_epi = myfont.render('reward_epi: ' + str(round(reward_epi,4)), False, pygame.Color('LightGray'))
    residual = myfont.render('residual: ' + str(np.round(character.state[0:2],2)), False, pygame.Color('LightGray'))
    border = myfont.render('border: ' + str(np.round(character.state[2:6],2)), False, pygame.Color('LightGray'))
    time_remaining = myfont.render('time remaining: ' + str(character.state[6]), False, pygame.Color('LightGray'))
    wind_compressed = myfont.render('wind_compressed: ' + str(np.round(character.state[7:],2)), False, pygame.Color('LightGray'))

    screen.blit(reward_step,(10,10))
    screen.blit(reward_epi,(10,25))
    screen.blit(residual,(10,55))
    screen.blit(border,(10,70))
    screen.blit(time_remaining,(10,85))
    screen.blit(wind_compressed,(10,100))

    # updating the window
    pygame.display.flip()
    clock.tick(10) #cycles per second
