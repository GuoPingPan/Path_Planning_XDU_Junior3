'''
    Astar 路径规划算法

    author: guopingpan
    email: panguoping02@gmail.com
            or 731061720@qq.com
    version:v0.2
'''

import numpy as np
import pygame

'''
    颜色定义
    
    path_color:  路径颜色,红色
    block_color: 障碍物颜色,黑色
    free_color:  空闲区域颜色,白色
    goal_color:  目标点颜色,绿色
    
    eightpoint:  以当前位置中心的可走范围（周围的八个点）
    
    fps: 构建路径显示帧率,实际生成路径过程是很快的，只是因为这里加了显示
'''
node_color = (255, 0, 0)
block_color = (0, 0, 0)
free_color = (255, 255, 255)
goal_color = (0, 255, 0)
path_color = (0, 155, 255)
begin_color = (0, 0, 255)

eightpoint = np.array([[-1, -1], [0, -1], [1, -1],
                       [-1,  0],          [1,  0],
                       [-1,  1], [0,  1], [1,  1]])

fps = 100

''' 节点类 '''
class Node:
    '''

    Args:
        pose(np.array): 节点的坐标
        father(Node):   父节点
        g_cost(float):  起点到当前节点的代价
        h_cost(float):  当前节点到终点的代价
        f_cost(float):  总代价

    Functions:
        __init__: 初始化代价为 0,父节点为 None
        update:   初次更新父节点
        update_with_newfather: 根据新的父节点进行更新

    Tips:
        f_cost 由 h_cost 和 g_cost 两部分组成
        其中：g_cost 为从起点到当前点所走路径的代价总和，并非起点到当前点的欧拉距离
             h_cost 为当前点到终点的欧拉距离

        使用 np.linalg.norm 来计算距离，可见对角线两点代价为 sqrt(2)，相邻两点代价为 1

    '''

    def __init__(self, pose):
        self.pose = pose
        self.father = None
        self.g_cost = 0
        self.h_cost = 0
        self.f_cost = 0

    def update(self, father, end):
        self.g_cost = father.g_cost + np.linalg.norm(father.pose - self.pose)
        self.h_cost = np.linalg.norm(self.pose - end)
        self.f_cost = self.g_cost + self.h_cost
        self.father = father

    def update_with_newfather(self, newfather):
        g_cost = newfather.g_cost + np.linalg.norm(newfather.pose - self.pose)
        if g_cost < self.g_cost:
            self.g_cost = g_cost
            self.f_cost = self.g_cost + self.h_cost
            self.father = newfather

''' Astar '''
class Astar:
    '''

    Args:
        init_pose(np.array):  起点
        end_pose(np.array):   终点
        openlist(list of Node): 允许走的点列表
        closedlist(np.array):  在添加障碍物时生成地图大小的障碍物矩阵。为 1 则障碍，为 0 则空闲

    Tips:
        这里 closedlist 使用 np.array 的形式大大加快了搜索的效率

    Functions:
        __init__:    初始化
        run:         运行主程序
        get_path:    获得路径
        display:     显示寻路过程
        add_block:   添加障碍物
        update_list: 更新 openlist
        find_goodNode_in_openlist: 在 openlist 中寻找最好点(代价最低点)
        is_in_openlist(return[Bool,index]): 判断当前节点是否在openlist中。在,则返回[True,index];否,则返回[False,-1]
    '''

    def __init__(self, Xmap, Ymap, init_pose, end_pose):
        self.map_size = [Xmap, Ymap]
        self.openlist = []
        self.closedlist = None
        self.init_pose = init_pose
        self.end_pose = end_pose
        init = Node(pose=self.init_pose)
        self.openlist.append(init)
        self.goodNode = None
        self.find_successfully = False

        ''' 使用pygame来展示全过程 '''
        pygame.init()
        self.window = pygame.display.set_mode((640, 640))
        self.grid = int(640 / Xmap)
        pygame.display.set_caption('Astar搜寻路径')
        self.clock = pygame.time.Clock()

    def run(self):

        while (len(self.openlist) != 0) and not (self.is_in_openlist(self.end_pose)[0]):
            self.find_goodNode_in_openlist()
            self.openlist.remove(self.goodNode)
            self.closedlist[self.goodNode.pose[0]][self.goodNode.pose[1]] = 1
            self.update_list(self.goodNode.pose)

            # 若想看算法的真实速度，请注释以下三条语句
            self.clock.tick(fps) # 这里频率可调，来显示刷新快慢
            self.display(node_color)
            pygame.display.flip()

        if (len(self.openlist) == 0):
            print("can't find the path")
            self.find_successfully = False
        else:
            print(f'path: {list(reversed(self.get_path()))}')
            self.find_successfully = True

    def quit(self):
        quit = True
        if self.find_successfully:
            self.display(path_color)
            pygame.display.flip()
        while quit:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit = False

    def get_path(self):
        path = []
        goodNode = self.goodNode
        while np.linalg.norm(goodNode.pose - self.init_pose) != 0:
            path.append(tuple(goodNode.pose))
            goodNode = goodNode.father
        return path

    def display(self,color):
        path = self.get_path()
        for node in path:
            pygame.draw.rect(self.window, color, (node[1] * self.grid, node[0] * self.grid, self.grid, self.grid))

    def find_goodNode_in_openlist(self):
        f_cost = 10000
        for Node in self.openlist:
            if Node.f_cost < f_cost:
                f_cost = Node.f_cost
                self.goodNode = Node

    def update_list(self, pose):
        pose_list = np.array([pose for i in range(8)], dtype=int) + eightpoint
        flag = [1 for i in range(8)]
        for i, pose in enumerate(pose_list):
            if pose[0] >= 0 and pose[0] < self.map_size[0] and pose[1] >= 0 and pose[1] < self.map_size[1]:
                if not (self.closedlist[pose[0]][pose[1]]):
                    flag[i] = 0
        # 当两个对角线的节点都为障碍物时，无法从两者中间穿过，即无法走对角线
        if flag[1] & flag[3] == 1:
            flag[0] = 1
        if flag[1] & flag[4] == 1:
            flag[2] = 1
        if flag[3] & flag[6] == 1:
            flag[5] = 1
        if flag[4] & flag[6] == 1:
            flag[7] = 1

        # 根据flag的值来更新openlist
        for i, signal in enumerate(flag):
            if signal == 1:
                continue
            else:
                yes, index = self.is_in_openlist(pose_list[i])
                if (yes):
                    self.openlist[index].update_with_newfather(self.goodNode)
                else:
                    node = Node(pose_list[i])
                    node.update(self.goodNode, self.end_pose)
                    self.openlist.append(node)

    def is_in_openlist(self, pose):
        index = -1
        for Node in self.openlist:
            index += 1
            if np.linalg.norm(pose - Node.pose) == 0:
                return [True, index]
        return [False, index]

    def add_block(self, block_pose):
        self.closedlist = block_pose

        # 将起点终点都强制设为可行区域
        if self.closedlist[self.init_pose[0]][self.init_pose[1]] == 1:
            self.closedlist[self.init_pose[0]][self.init_pose[1]] = 0
        if self.closedlist[self.end_pose[0]][self.end_pose[1]] == 1:
            self.closedlist[self.end_pose[0]][self.end_pose[1]] = 0

        # 画出障碍物地图
        for i in range(block_pose.shape[0]):
            for j in range(block_pose.shape[1]):
                if (block_pose[i][j] == 1):
                    pygame.draw.rect(self.window, block_color, ( j * self.grid,i * self.grid, self.grid, self.grid))
                else:
                    pygame.draw.rect(self.window, free_color, ( j * self.grid,i * self.grid, self.grid, self.grid))

        # 画出起点和终点
        pygame.draw.rect(self.window, begin_color,( self.init_pose[1] * self.grid,self.init_pose[0] * self.grid, self.grid, self.grid))
        pygame.draw.rect(self.window, goal_color,( self.end_pose[1] * self.grid,self.end_pose[0] * self.grid, self.grid, self.grid))


''' main '''
import time # 引入time模块计时间，和blocklist为Node的版本作比较
# np.random.seed(3)

if __name__ == '__main__':
    map_size = [20, 20]
    astar = Astar(map_size[0], map_size[1], np.array([0, 0]), np.array([18, 18]))#18,6
    #block_pose = np.random.randint(0, 2, size=map_size, dtype=int)
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
                          ],dtype=int)
    astar.add_block(block_pose)
    t1 = time.time()
    astar.run()
    print(f'use time: {time.time() - t1} s')
    print(f'这里是由于显示消耗了时间，可以根据 run 程序中的注释来去除显示模块，看看算法的真实运行速度')
    astar.quit()