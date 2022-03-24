from sys import maxsize
import numpy as np
import pygame
import time

path_color = (255, 0, 0)
block_color = (0, 0, 0)
free_color = (255, 255, 255)
goal_color = (0, 255, 0)

fps = 100

class Node:
    def __init__(self, position):
        self.position = position
        self.parent = None
        self.h = maxsize

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

    def init_closelist(self):
        closelist=[]
        for i in range(self.row):
            for j in range(self.col):
                closelist.append(self.map[i][j])
        return closelist

    def get_neighbor(self, node):
        neighbor = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                if node.position[0] + i >= 0 and node.position[0] + i < 20:
                    if node.position[1] + j >= 0 and node.position[1] + j < 20:
                        if self.Map[node.position[0] + i][node.position[1] + j] == 0:
                            neighbor.append(self.map[node.position[0] + i][node.position[1] + j])
        return neighbor

    def add_block(self, block_pose,start_node,end_node):
        self.Map = block_pose
        self.start_node=start_node
        self.end_node=end_node

        # 将起点终点都强制设为可行区域
        if self.Map[self.start_node[0]][self.start_node[1]] == 1:
            self.Map[self.start_node[0]][self.start_node[1]] = 0
        if self.Map[self.end_node[0]][self.end_node[1]] == 1:
            self.Map[self.end_node[0]][self.end_node[1]] = 0

        # 画出障碍物地图
        for i in range(block_pose.shape[0]):
            for j in range(block_pose.shape[1]):
                if (block_pose[i][j] == 1):
                    pygame.draw.rect(self.window, block_color, (j * self.grid, i * self.grid, self.grid, self.grid))
                else:
                    pygame.draw.rect(self.window, free_color, (j * self.grid, i * self.grid, self.grid, self.grid))

        # 画出起点和终点
        pygame.draw.rect(self.window, path_color,
                         (self.start_node[1] * self.grid, self.start_node[0] * self.grid, self.grid, self.grid))
        pygame.draw.rect(self.window, goal_color,
                         (self.end_node[1] * self.grid, self.end_node[0] * self.grid, self.grid, self.grid))

    def cost(self, x_node, y_node):
        if self.Map[x_node.position[0]][x_node.position[1]] == 1 or \
                self.Map[y_node.position[0]][y_node.position[1]] == 1:
            return maxsize
        return np.linalg.norm(x_node.position - y_node.position)

class Dijkstra:
    def __init__(self, Xmap, Ymap, start_node, end_node, maps):
        self.Xmap = Xmap
        self.Ymap = Ymap
        self.map = maps
        self.start_node = start_node
        self.end_node = end_node
        self.openlist = []
        self.closelist =self.map.init_closelist()
        self.Map = None

        self.window = pygame.display.set_mode((640, 640))
        self.grid = int(640 / Xmap)
        pygame.display.set_caption('Dstar搜寻路径')
        self.clock = pygame.time.Clock()

    def search(self):
        if len(self.closelist) == 0:
            return -1
        x=min(self.closelist,key=lambda a: a.h)
        self.openlist.append(x)
        self.closelist.remove(x)
        neighbor_x = self.map.get_neighbor(x)
        for y in neighbor_x:
            if y.h>x.h+self.map.cost(x,y):
                y.parent=x
                y.h=x.h+self.map.cost(x,y)

    def run(self):
        self.closelist[self.start_node.position[0]*self.Ymap+self.start_node.position[1]].h=0
        while True:
            self.search()
            if self.end_node in self.openlist:
                break
            # 若想看算法的真实速度，请注释以下三条语句
            self.clock.tick(fps)  # 这里频率可调，来显示刷新快慢
            self.display()
            pygame.display.flip()

        if (len(self.openlist) == 0):
            print("can't find the path")
        else:
            print(f'path: {list(reversed(self.get_path()))}')

    def get_path(self):
        path = []
        # node = self.end_node.parent
        node=self.openlist[-1]
        while node is not None and np.linalg.norm(node.position - self.start_node.position) != 0:
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
    dijkstra = Dijkstra(map_size[0], map_size[1], m.map[0][0], m.map[10][10], m)
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
    m.add_block(block_pose,np.array([0,0]),np.array([10,10]))
    t1 = time.time()
    dijkstra.run()
    print(f'use time: {time.time() - t1} s')
    dijkstra.quit()




