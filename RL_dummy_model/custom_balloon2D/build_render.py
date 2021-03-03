import pygame, sys

def build_render(state, target, size_x, size_z, t, world_name, window_size, train_or_test):
    # general setup
    pygame.init()
    clock = pygame.time.Clock()

    # setting up the main window
    res = 10
    screen_width = size_x * res
    screen_height = size_z * res
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('balloon2d')

    # game rectangles (orgin is always on the top left, so increase y to go down)
    size_obs_x = window_size*2*res
    size_obs_y = size_z*2*res
    pos_obs = [state[0]*res, size_z/2*res]
    rec_obs = pygame.Rect(pos_obs[0] - size_obs_x/2, pos_obs[1] - size_obs_y/2, size_obs_x, size_obs_y)

    size_balloon = 1*res
    pos_balloon = [state[0]*res, (size_z - state[1])*res]
    rec_balloon = pygame.Rect(pos_balloon[0] - size_balloon/2, pos_balloon[1] - size_balloon/2, size_balloon, size_balloon)

    size_target = 1*res
    pos_target = [target[0]*res, (size_z - target[1])*res]
    rec_target = pygame.Rect(pos_target[0] - size_target/2, pos_target[1] - size_target/2, size_target, size_target)

    # handling input
    for event in pygame.event.get():
        if event.type == pygame.QUIT: #does user click closing button
            pygame.quit()
            sys.exit()

    # generate background
    screen.fill(pygame.Color('grey12'))
    pygame.draw.rect(screen, pygame.Color('DarkSlateGray'), rec_obs) #draw window
    bg = pygame.image.load('data/' + train_or_test + '/image/' + world_name + '.png')
    bg = pygame.transform.scale(bg, (screen_width, screen_height))
    screen.blit(bg, (0, 0))

    # visuals
    pygame.draw.ellipse(screen, pygame.Color('LightGray'), rec_balloon)
    pygame.draw.ellipse(screen, pygame.Color('FireBrick'), rec_target)

    # writing which step we are rendering
    myfont = pygame.font.SysFont('Arial', 15, bold = True)
    textsurface = myfont.render('time remaining: ' + str(t), False, pygame.Color('LightGray'))
    screen.blit(textsurface,(10,10))

    # updating the window
    pygame.display.flip()
    clock.tick(60) #cycles per second
