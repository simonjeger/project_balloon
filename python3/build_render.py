import numpy as np
import torch
import pygame, sys
from sys import exit

from visualize_world import visualize_world
from preprocess_wind import squish

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)


class render():

    def __init__(self, size_x, size_y, size_z):
        self.render_ratio = int(yaml_p['unit_xy'] / yaml_p['unit_z'])

        self.size_x = (size_x - 1)*self.render_ratio
        self.size_y = (size_y - 1)*self.render_ratio
        self.size_z = size_z - 1

    def make_render(self, character, action, reward_step, reward_epi, radius_xy, radius_z, train_or_test, roll_out, tss, load_screen):
        # general setup
        pygame.init()
        clock = pygame.time.Clock()

        # setting up the main window
        res = 1.5*100/self.size_z #int(100/self.size_z)
        screen_width = int(3*self.size_z*res)
        screen_height = int((2*self.size_z)*res + screen_width)
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption('balloon3d')

        c_background = (34,42,53)
        screen.fill(c_background)

        # handling input
        for event in pygame.event.get():
            if event.type == pygame.QUIT: #does user click closing button
                pygame.quit()
                sys.exit()

        # define window size
        window_size = character.ae.window_size

        # read in wind_map
        world = torch.load(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/render/world.pt')

        # generate the three windows
        visualize_world(world, character.position, character.ceiling)

        for dim in ['xy', 'xz', 'yz']:
            self.display_movement(dim, screen, screen_width, screen_height, c_background, window_size, radius_xy, radius_z, res, character, action, roll_out)

        # text
        myfont = pygame.font.SysFont('Arial', 10, bold=False)
        hour = int(tss/60/60)
        minute = int(tss/60 - hour*60)
        second = int(tss - hour*60*60 - minute*60)
        t_tss = myfont.render('UTC time: ' + str(hour) + ':' + str(minute) + ':' + str(second), False, pygame.Color('LightGray'))

        t_reward_step = myfont.render('reward_step: ' + '{:.3f}'.format(reward_step), False, pygame.Color('LightGray'))
        t_reward_epi = myfont.render('reward_epi: ' + '{:.3f}'.format(reward_epi), False, pygame.Color('LightGray'))
        t_action = myfont.render('action: ' + str(np.round(character.action,3)), False, pygame.Color('LightGray'))
        t_diameter = myfont.render('diameter: ' + str(np.round(character.diameter,2)), False, pygame.Color('LightGray'))
        t_battery_level = myfont.render('battery_level: ' + str(np.round(character.battery_level,2)), False, pygame.Color('LightGray'))
        t_time = myfont.render('time: ' + str(np.round((character.T - character.t)/60,2)) + ' / ' + str(np.round(character.T/60,2)) + ' min', False, pygame.Color('LightGray'))
        t_residual = myfont.render('residual: ' + str([round(num, 3) for num in character.state[0:3].tolist()]), False, pygame.Color('LightGray'))
        t_velocity = myfont.render('velocity: ' + str([round(num, 3) for num in character.state[3:6].tolist()]), False, pygame.Color('LightGray'))

        t_rel_pos = myfont.render('rel_pos: ' + str([round(num, 3) for num in character.state[6:10].tolist()]), False, pygame.Color('LightGray'))
        t_measurement = myfont.render('measurement: ' + str([round(num, 3) for num in character.state[10:12].tolist()]), False, pygame.Color('LightGray'))
        t_world_compressed = myfont.render('world_compressed: ' + str([round(num, 3) for num in character.state[12:12+yaml_p['bottleneck']*2].tolist()]), False, pygame.Color('LightGray'))
        t_esterr = myfont.render('esterror_pos: ' + str([round(num, 3) for num in [character.esterror_pos, character.esterror_vel, character.esterror_wind]]), False, pygame.Color('LightGray'))

        start_text = 2*self.size_z*res
        space_text = 15
        screen.blit(t_tss,(space_text,start_text+1*space_text))
        screen.blit(t_reward_step,(space_text,start_text+3*space_text))
        screen.blit(t_reward_epi,(space_text,start_text+4*space_text))
        screen.blit(t_action,(space_text,start_text+5*space_text))
        screen.blit(t_diameter,(space_text,start_text+6*space_text))
        screen.blit(t_battery_level,(space_text,start_text+7*space_text))
        screen.blit(t_time,(space_text,start_text+8*space_text))
        if yaml_p['mode'] != 'game':
            screen.blit(t_residual,(space_text,start_text+10*space_text))
            screen.blit(t_velocity,(space_text,start_text+11*space_text))
            screen.blit(t_rel_pos,(space_text,start_text+12*space_text))
            screen.blit(t_measurement,(space_text,start_text+13*space_text))
            screen.blit(t_world_compressed,(space_text,start_text+14*space_text))
            screen.blit(t_esterr,(space_text,start_text+16*space_text))

        if load_screen:
            t_load_screen = myfont.render('PRESS NUMBER KEY TO START GAME', False, pygame.Color('White'))
            screen.blit(t_load_screen,(screen_width/2,start_text+15*space_text))

        # updating the window
        pygame.display.flip()
        clock.tick(max(1,1/yaml_p['delta_t'])) #cycles per second

    def display_movement(self, dim, screen, screen_width, screen_height, c_background, window_size, radius_xy, radius_z, res, character, action, roll_out):
        if dim == 'xz':
            i1 = 0
            i2 = 2
            size_1 = self.size_x
            size_2 = self.size_z
            dist_to_top = self.size_z
            dist_to_bottom = 2*self.size_z
            position_1 = character.position[i1]*self.render_ratio
            position_2 = character.position[i2]
            target_1 = character.target[i1]*self.render_ratio
            target_2 = character.target[i2]
            ceiling = character.ceiling

        if dim == 'yz':
            i1 = 1
            i2 = 2
            size_1 = self.size_y
            size_2 = self.size_z
            dist_to_top = 0
            dist_to_bottom = self.size_z
            position_1 = character.position[i1]*self.render_ratio
            position_2 = character.position[i2]
            target_1 = character.target[i1]*self.render_ratio
            target_2 = character.target[i2]
            ceiling = character.ceiling

        if dim == 'xy':
            i1 = 0
            i2 = 1
            size_1 = self.size_x
            size_2 = self.size_y
            dist_to_top = 2*self.size_z
            dist_to_bottom = screen_height/res
            position_1 = character.position[i1]*self.render_ratio
            position_2 = character.position[i2]*self.render_ratio
            target_1 = character.target[i1]*self.render_ratio
            target_2 = character.target[i2]*self.render_ratio

        window_size = window_size*self.render_ratio

        # colors
        c_ceiling = (150,150,150,150)
        c_balloon = (173,29,0)
        c_action = (111,21,2)
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
        bg = pygame.image.load(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/render/render_' + dim + '.png')
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
            size_ceil_y = max((size_2-ceiling)*res,1) #otherwise it can't render it for very small ceiling
            pos_ceil = [0, dist_to_top*res]
            rec_ceil = pygame.Rect(pos_ceil[0], pos_ceil[1], size_ceil_x, size_ceil_y)
            shape_surf = pygame.Surface(pygame.Rect(rec_ceil).size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, c_ceiling, shape_surf.get_rect())
            screen.blit(shape_surf, rec_ceil)

        if yaml_p['mode'] != 'game':
            # write and display observing box
            size_obs_x = (window_size*2+1*self.render_ratio)*res
            if dim != 'xy':
                size_obs_y = (dist_to_bottom - dist_to_top)*res
                pos_obs = [(int(position_1/self.render_ratio)*self.render_ratio + offset_1 - window_size)*res, dist_to_top*res]
            else:
                size_obs_y = (window_size*2+1*self.render_ratio)*res
                pos_obs = [(int(position_1/self.render_ratio)*self.render_ratio + offset_1 - window_size)*res, (int((dist_to_bottom - position_2)/self.render_ratio)*self.render_ratio - offset_2 - window_size)*res]
            rec_obs = pygame.Rect(pos_obs[0], pos_obs[1], size_obs_x, size_obs_y)

            shape_surf = pygame.Surface(pygame.Rect(rec_obs).size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, c_window, shape_surf.get_rect())
            screen.blit(shape_surf, rec_obs)

        # write path
        path = []
        if dim != 'xy':
            for i in character.path:
                path.append(((i[i1]*self.render_ratio+offset_1)*res, (dist_to_bottom-i[i2])*res))
        else:
            for i in character.path:
                path.append(((i[i1]*self.render_ratio+offset_1)*res, (dist_to_bottom-i[i2]*self.render_ratio-offset_2)*res))

        if len(roll_out) > 0:
            path_roll_out = []
            if dim != 'xy':
                for i in roll_out:
                    path_roll_out.append(((i[i1]*self.render_ratio+offset_1)*res, (dist_to_bottom-i[i2])*res))
            else:
                for i in roll_out:
                    path_roll_out.append(((i[i1]*self.render_ratio+offset_1)*res, (dist_to_bottom-i[i2]*self.render_ratio-offset_2)*res))

        if action is not None:
            if dim == 'xz':
                pos_y = int(np.clip(character.position[1],0,character.size_y-1))
                action_set = (ceiling - character.world[0,:,pos_y,0])*action + character.world[0,:,pos_y,0]

            if dim == 'yz':
                pos_x = int(np.clip(character.position[0],0,character.size_x-1))
                action_set = (ceiling - character.world[0,pos_x,:,0])*action + character.world[0,pos_x,:,0]

            if dim != 'xy':
                array_2 = np.arange(0,size_1+1,1)
                line_action = []
                for i in range(len(action_set)):
                    line_action.append(((array_2[i]*self.render_ratio + offset_1)*res, (dist_to_bottom-action_set[i])*res))
                pygame.draw.lines(screen, c_action, False, line_action, 1)

        # write balloon
        size_balloon = 4*res
        if dim != 'xy':
            pos_balloon = [(position_1 + offset_1)*res, (dist_to_bottom - position_2)*res]
        else:
            pos_balloon = [(position_1 + offset_1)*res, (dist_to_bottom - position_2 - offset_2)*res]
        rec_balloon = pygame.Rect(pos_balloon[0] - size_balloon/2, pos_balloon[1] - size_balloon/2, size_balloon, size_balloon)

        # write target
        size_target = 4*res
        if dim != 'xy':
            pos_target = [(target_1 + offset_1)*res, (dist_to_bottom - target_2)*res]
        else:
            pos_target = [(target_1 + offset_1)*res, (dist_to_bottom - target_2 - offset_2)*res]
        rec_target = pygame.Rect(pos_target[0] - size_target/2, pos_target[1] - size_target/2, size_target, size_target)

        # path
        if (len(roll_out) > 0) & (yaml_p['mode'] != 'game'):
            pygame.draw.lines(screen, c_path_roll_out, False, path_roll_out, 1)

        if len(path) > 1:
            pygame.draw.lines(screen, c_path, False, path, 1)

        # balloon
        cv = 100
        pygame.draw.ellipse(screen, c_balloon, rec_balloon)

        # draw target (dense and transparent)
        pygame.draw.ellipse(screen, c_target_center, rec_target)

        if dim != 'xy':
            r_1 = radius_xy*res
            r_2 = radius_z*res
        else:
            r_1 = radius_xy*res
            r_2 = radius_xy*res

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
        pygame.draw.circle(screen, c_balloon, position_balloon_mm, 1)


        mm_target = [target_1/size_1*length_mm, height_mm - target_2/size_2*height_mm]
        position_target_mm = [position_mm[0]+mm_target[0], position_mm[1]+mm_target[1]]
        pygame.draw.circle(screen, c_target_radius, position_target_mm, 1)

        # overlay everything that overlaps from the xy plane
        if dim == 'xy':
            pygame.draw.rect(screen, c_background, pygame.Rect(0, 0, screen_width, dist_to_top*res))
