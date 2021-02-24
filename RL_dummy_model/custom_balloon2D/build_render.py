import pygame, sys

def build_render(state, target, size_x, size_z, t):
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
    bg = pygame.image.load("wind_map.png")
    #bg = pygame.transform.scale(bg, (screen_width, screen_height))
    #screen.blit(bg, (0, 0))

    # visuals
    pygame.draw.ellipse(screen, pygame.Color('LightGray'), rec_balloon)
    pygame.draw.ellipse(screen, pygame.Color('IndianRed'), rec_target)

    # writing which step we are rendering
    myfont = pygame.font.SysFont('Arial', 15)
    textsurface = myfont.render('remaining time: ' + str(t), False, pygame.Color('LightGray'))
    screen.blit(textsurface,(0,0))

    # updating the window
    pygame.display.flip()
    clock.tick(15) #cycles per second
