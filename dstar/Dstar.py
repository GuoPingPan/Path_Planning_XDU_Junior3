from sys import maxsize
import numpy as np
import pygame
import time

node_color = (255, 0, 0)
block_color = (0, 0, 0)
reblock_color=(50,50,50)
free_color = (255, 255, 255)
goal_color = (0, 255, 0)
begin_color = (0,0,255)
path_color = (0, 155, 255)
repath_color=(255,255,0)

fps = 100


class Node:
    def __init__(self, position):
        self.position = position
        self.state = 'new'
        self.parent = None
        self.h = 0
        self.k = 0
        self.obstacle=False


class Map(object):
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.map = self.init_map()
        pygame.init()
        self.window = pygame.display.set_mode((640, 640))
        self.grid = int(640 / 20)
        pygame.display.set_caption('Dstar搜寻路径')
        self.clock = pygame.time.Clock()

    def init_map(self):
        # 初始化map
        map_list = []
        for i in range(self.row):
            tmp = []
            for j in range(self.col):
                tmp.append(Node(position=np.array([i, j])))
            map_list.append(tmp)
        return map_list

    def add_block(self, block_pose, start_node, end_node):
        self.Map = block_pose
        self.start_node = start_node
        self.end_node = end_node

        # 将起点终点都强制设为可行区域
        if self.Map[self.start_node[0]][self.start_node[1]] == 1:
            self.Map[self.start_node[0]][self.start_node[1]] = 0
        if self.Map[self.end_node[0]][self.end_node[1]] == 1:
            self.Map[self.end_node[0]][self.end_node[1]] = 0

        # 画出障碍物地图
        for i in range(block_pose.shape[0]):
            for j in range(block_pose.shape[1]):
                if (block_pose[i][j] == 1):
                    self.map[i][j].obstacle = True
                    pygame.draw.rect(self.window, block_color, (j * self.grid, i * self.grid, self.grid, self.grid))
                else:
                    pygame.draw.rect(self.window, free_color, (j * self.grid, i * self.grid, self.grid, self.grid))

        # 画出起点和终点
        pygame.draw.rect(self.window, begin_color,
                         (self.start_node[1] * self.grid, self.start_node[0] * self.grid, self.grid, self.grid))
        pygame.draw.rect(self.window, goal_color,
                         (self.end_node[1] * self.grid, self.end_node[0] * self.grid, self.grid, self.grid))

    # def add_obstacle(self,obstacle_list):
    #     for obstacle in obstacle_list:
    #         self.Map[obstacle[0]][obstacle[1]] = 1
    #         self.map[obstacle[0]][obstacle[1]].obstacle=True
    #         pygame.draw.rect(self.window, reblock_color, (obstacle[1] * self.grid, obstacle[0] * self.grid, self.grid, self.grid))
    #         pygame.display.flip()

    def add_obstacle(self):
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                print('mouse',event.pos)
                position=self.get_position(event.pos)
                print(position)
                self.Map[position[0]][position[1]] = 1
                self.map[position[0]][position[1]].obstacle=True
                pygame.draw.rect(self.window, reblock_color, (position[1] * self.grid, position[0] * self.grid, self.grid, self.grid))
                pygame.display.flip()
            if event.type==pygame.MOUSEBUTTONUP:
                return 1

    def get_position(self,pos):
        position=[0,0]
        for i in range(20):
            if pos[0]>=i*32 and pos[0]<(i+1)*32:
                position[1]=i
            if pos[1] >= i * 32 and pos[1] < (i + 1) * 32:
                position[0] = i
        return position

    def get_neighbor(self, node):
        neighbor = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                if node.position[0] + i >= 0 and node.position[0] + i < 20:
                    if node.position[1] + j >= 0 and node.position[1] + j < 20:
                        neighbor.append(self.map[node.position[0] + i][node.position[1] + j])
        return neighbor

    def cost(self, x_node, y_node):
        if self.Map[x_node.position[0]][x_node.position[1]] == 1 or \
                self.Map[y_node.position[0]][y_node.position[1]] == 1:
            return 1000
        if (y_node.position[0] == x_node.position[0] + 1) and (y_node.position[1] == x_node.position[1] + 1) \
                and (self.Map[x_node.position[0] + 1][x_node.position[1]] == 1) and (
                self.Map[x_node.position[0]][x_node.position[1] + 1] == 1):
            return 1000
        if (y_node.position[0] == x_node.position[0] - 1) and (y_node.position[1] == x_node.position[1] - 1) \
                and (self.Map[x_node.position[0] - 1][x_node.position[1]] == 1) and (
                self.Map[x_node.position[0]][x_node.position[1] - 1] == 1):
            return 1000
        if (y_node.position[0] == x_node.position[0] - 1) and (y_node.position[1] == x_node.position[1] + 1) \
                and (self.Map[x_node.position[0] - 1][x_node.position[1]] == 1) and (
                self.Map[x_node.position[0]][x_node.position[1] + 1] == 1):
            return 1000
        if (y_node.position[0] == x_node.position[0] + 1) and (y_node.position[1] == x_node.position[1] - 1) \
                and (self.Map[x_node.position[0] - 1][x_node.position[1]] == 1) and (
                self.Map[x_node.position[0] + 1][x_node.position[1]] == 1):
            return 1000
        return np.linalg.norm(x_node.position - y_node.position)

class Dstar:
    def __init__(self, Xmap, Ymap, start_node, end_node, maps):
        self.Xmap = Xmap
        self.Ymap = Ymap
        self.map = maps
        self.start_node = start_node
        self.end_node = end_node
        self.openlist = []
        self.closelist = []
        self.Map = None

        self.window = pygame.display.set_mode((640, 640))
        self.grid = int(640 / Xmap)
        pygame.display.set_caption('Dstar搜寻路径')
        self.clock = pygame.time.Clock()

    def search(self):
        # print(len(self.openlist))
        if len(self.openlist) == 0:
            return -1
        x = min(self.openlist, key=lambda a: a.k)
        k_old = self.get_kmin()
        # print('x', x.position, x.h, k_old)
        self.openlist.remove(x)
        self.closelist.append(x)
        if x!=self.end_node and x!=self.start_node and x.obstacle!=True:
            pygame.draw.rect(self.window, node_color,(x.position[1] * self.grid, x.position[0] * self.grid, self.grid, self.grid))
        x.state = 'close'
        neighbor_x = self.map.get_neighbor(x)
        if k_old < x.h:
            for y in neighbor_x:
                if y.h <= k_old and x.h > y.h + self.map.cost(y, x):
                    x.parent = y
                    x.h = y.h + self.map.cost(y, x)
        elif k_old == x.h:
            for y in neighbor_x:
                if (y.state == 'new') or (y.parent == x and y.h != x.h + self.map.cost(x, y)) \
                        or (y.parent != x and y.h > x.h + self.map.cost(x, y)):
                    y.parent = x
                    self.Insert(y, x.h + self.map.cost(x, y))
        else:
            for y in neighbor_x:
                if (y.state == 'new') or (y.parent == x and y.h != x.h + self.map.cost(x, y)):
                    y.parent = x
                    self.Insert(y, x.h + self.map.cost(x, y))
                elif (y.parent != x and y.h > x.h + self.map.cost(x, y)):
                    self.Insert(x, x.h)
                else:
                    if y.parent != x and x.h > y.h + self.map.cost(y, x) and y.state == 'close' \
                            and y.h > k_old:
                        self.Insert(y, y.h)
        return self.get_kmin()

    def get_kmin(self):
        if len(self.openlist) == 0:
            return -1
        k_min = min([x.k for x in self.openlist])
        return k_min

    def run(self):
        self.openlist.append(self.end_node)
        while True:
            self.search()
            if self.start_node in self.closelist:
                break
            # # 若想看算法的真实速度，请注释以下三条语句
            self.clock.tick(fps)  # 这里频率可调，来显示刷新快慢
            # self.display()
            pygame.display.flip()

        if (len(self.openlist) == 0):
            print("can't find the path")
        else:
            print(f'path: {list((self.get_path()))}')
            self.clock.tick(fps)  # 这里频率可调，来显示刷新快慢
            self.display()
            pygame.display.flip()

    def obstacle(self):
        # obstacle_list=[(1,1),(2,2),(4,3),(5,4)]
        # self.map.add_obstacle(obstacle_list)
        while True:
            flag=self.map.add_obstacle()
            if flag==1:
                t1 = time.time()
                node=self.start_node
                while node != self.end_node:
                    if node.parent.obstacle == True:
                        self.modify(node)
                        continue
                    pygame.draw.rect(self.window, repath_color,
                                     (node.position[1] * self.grid, node.position[0] * self.grid, self.grid, self.grid))
                    pygame.display.flip()
                    node = node.parent
                print(f'path: {list((self.get_path()))}')
                self.clock.tick(fps)  # 这里频率可调，来显示刷新快慢
                path = self.get_path()
                for node in path:
                    pygame.draw.rect(self.window, repath_color, (node[1] * self.grid, node[0] * self.grid, self.grid, self.grid))
                pygame.display.flip()
                print(f'use time: {time.time() - t1} s')

    def modify(self, node):
        self.Insert(node.parent,node.parent.h)
        neighbor=self.map.get_neighbor(node.parent)
        for n in neighbor:
            if n.state=='close':
                self.Insert(n, n.h+self.map.cost(node.parent,n))
        while True:
            k_min = self.search()
            # print(k_min,node.h)
            if node.h <= k_min:
                break

    def Insert(self, x, h_new):
        if x.state == 'new':
            x.k = h_new
        elif x.state == 'open':
            x.k = min(x.k, h_new)
        elif x.state == 'close':
            x.k = min(x.h, h_new)
        x.h = h_new
        x.state = 'open'
        self.openlist.append(x)

    def get_path(self):
        path = []
        node = self.start_node.parent
        # node=self.closelist[-1]
        while node is not None and np.linalg.norm(node.position - self.end_node.position) != 0:
            path.append(tuple(node.position))
            node = node.parent
        return path

    def display(self):
        path = self.get_path()
        for node in path:
            pygame.draw.rect(self.window, path_color, (node[1] * self.grid, node[0] * self.grid, self.grid, self.grid))

    def quit(self):
        quit = True
        self.display()
        while quit:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit = False

if __name__ == '__main__':
    map_size = [20, 20]
    m = Map(20, 20)
    dstar = Dstar(map_size[0], map_size[1], m.map[0][0], m.map[19][19], m)
    block_pose = np.array([[0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
                           [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
                           [0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
                           [0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0],
                           [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0],
                           [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0],
                           [0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                           [0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                           [0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                           [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                           [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
                           [0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
                           ], dtype=int)
    m.add_block(block_pose,np.array([0,0]),np.array([19,19]))
    t1 = time.time()
    dstar.run()
    print(f'use time: {time.time() - t1} s')
    dstar.obstacle()
    dstar.quit()




