'''
    RRT(Rapidly-exploring Random Trees)
    随机快速扩展树

    author: guopingpan<panguoping02@gmail.com>
    version: v2.0

    This project is using rrt algorithm to deal with path planning jobs.
    Dynamic means you can use mouse to click the map and add some block.
    If there is another path then it can find out,
        but if there is not another path then it will be trapped and close it by yourself.
'''
import numpy as np
import pygame

'''
    颜色定义

    node_color:  节点颜色,红色
    block_color: 障碍物颜色,黑色
    free_color:  空闲区域颜色,白色
    goal_color:  目标点颜色,绿色
    begin_color: 起点颜色,深蓝色
    path_color:  路径颜色,浅蓝色
    new_block_color: 新添障碍物的颜色，深灰色
    fps: 构建路径显示帧率,实际生成路径过程是很快的，只是因为这里加了显示
'''

node_color = (255, 0, 0)
block_color = (0, 0, 0)
free_color = (255, 255, 255)
goal_color = (0, 255, 0)
begin_color = (0,0,255)
path_color = (0, 155, 255)
new_block_color = (76, 80, 82)
fps = 100


class Node:
    '''
        节点类

    Args:
        pose(np.array): 节点位置坐标
        father(Node):   父节点
        childs(List of Node): 孩子节点列表

    Functions:
        add_child:    添加孩子节点
        remove_child: 删除孩子节点

    '''
    def __init__(self,pose,father=None):
        self.pose = pose
        self.father = father
        self.childs = []

    def add_child(self,node):
        self.childs.append(node)
    def remove_child(self,node):
        self.childs.remove(node)

class RRT:
    '''
        搜索树

    Args:
        value(type)
        ----------------------------------------
        map_size(int,int):   地图尺寸
        root(Node):       根结点(亦为起始节点)
        end_node(Node):   终止节点
        end_pose(pose):   终止节点位置坐标
        path(list of tuple):       最终路径

        epsilon(float):    epsilon-贪心算法阈值

        block_list(np.array):  障碍物列表
        closed_list(list of Node): 存放走过的节点的列表

        out: 退出程序指示

    Tips:
        建立closed_list来存放节点，用于完成树的遍历，空间换时间

    Functions:
        run: 运行主程序
        search_new_pose_and_near_node: 遍历closedlist,找到距离next_pose最近点作为near_pose,
                                       通过near_pose和next_pose确定new_pose
            - update_when_collision:   在search_new_pose_and_near_node中被调用，
                                       判断new_pose的合法性，选择合适的new_pose
        get_path:  回溯获得路径
        add_block: 添加障碍物地图
        quit+display: 保持显示，直至收到退出指令
        need_modify:  当添加的障碍物影响到了原来的路径的时候进行修正
        get_mouse_position: 将鼠标点击的位置转化为地图中的坐标位置

    '''
    def __init__(self,X,Y,start,end,epsilon=0.3):
        self.map_size = (X,Y)
        self.root = Node(pose=start)
        self.end_pose = end
        self.block_list = None
        self.end_node = None
        self.closed_list = [self.root]
        self.epsilon = epsilon
        self.path = []
        self.out = False

        # display the process
        pygame.init()
        self.window = pygame.display.set_mode((640, 640))
        self.grid = int(640 / X)
        pygame.display.set_caption('RRT搜寻路径')
        self.clock = pygame.time.Clock()


    def run(self):

        while True and not self.out:
            while self.end_node is None:
                # epsilon-贪心算法: p<epsilon，随机选取下一个点；p>epsilon，选取终点为下一个点
                p = np.random.rand()
                if p < self.epsilon:
                    next_pose = np.random.randint(0,20,size=2)
                else:
                    next_pose = np.copy(self.end_pose)

                # 搜索new_pose和near_node
                success,new_pose,near_node = self.search_new_pose_and_near_node(next_pose)

                # 搜索到end_pose，成功找到路径
                if np.linalg.norm(new_pose-self.end_pose) == 0:
                    self.end_node = Node(pose=self.end_pose,father=near_node)
                    break

                # 搜索成功，生成树枝
                else:
                    if success:
                        new_node = Node(pose=new_pose,father=near_node)
                        self.block_list[new_pose[0]][new_pose[1]] = 1
                        self.closed_list.append(new_node)
                        near_node.add_child(new_node)
                        pygame.draw.rect(self.window, node_color, (new_pose[1] * self.grid, new_pose[0] * self.grid, self.grid, self.grid))

                # display the process
                self.clock.tick(fps) # 这里频率可调，来显示刷新快慢
                pygame.display.flip()

            self.quit()


    def need_modify(self,position):
        # print(self.end_node)
        node = self.end_node.father
        while node is not self.root and np.linalg.norm(node.pose - position) != 0:
            node = node.father

        if node is not self.root:
            self.closed_list.remove(node)
            bfs = [node]
            cut = []
            while len(bfs) != 0:
                node = bfs.pop(0)# pop默认最后一个
                cut.append(node)
                for child in node.childs:
                    bfs.append(child)
            # print(cut)
            node = cut.pop(0)
            node.father.remove_child(node)

            for node in cut:
                # print(node)
                self.closed_list.remove(node)
                self.block_list[node.pose[0]][node.pose[1]] = 0
                pygame.draw.rect(self.window, free_color,(node.pose[1] * self.grid, node.pose[0] * self.grid, self.grid, self.grid))

            self.end_node = None
            return True

        return False

    def get_mouse_postion(self,pose):
        print(pose)
        return np.array([int(pose[1]/self.grid),int(pose[0]/self.grid)])

    def get_path(self):
        node = self.end_node.father
        path = []
        while node is not self.root:
            path.append((node.pose[0],node.pose[1]))
            node = node.father
        return list(reversed(path))

    def quit(self):
        quit = True
        self.path = self.get_path()
        print(f'path:{self.path}')
        self.display()
        pygame.display.flip()
        while quit:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit = False
                    self.out = True
                if event.type == pygame.MOUSEBUTTONDOWN:
                    position = self.get_mouse_postion(event.pos)
                    self.block_list[position[0]][position[1]] = 1
                    pygame.draw.rect(self.window, new_block_color,(position[1] * self.grid, position[0] * self.grid, self.grid, self.grid))
                    pygame.display.flip()
                    if quit:
                        if self.need_modify(position):
                            # print('yes')
                            pygame.display.flip()
                            quit = False


    def display(self):
        for pose in self.path:
            pygame.draw.rect(self.window, path_color,(pose[1] * self.grid, pose[0] * self.grid, self.grid, self.grid))

    def search_new_pose_and_near_node(self,next_pose):
        min_dist = 10000
        near_node = None

        for node in self.closed_list:
            dist = np.linalg.norm(node.pose-next_pose)
            if dist < min_dist:
                min_dist = dist
                near_node = node

        near_pose = np.copy(near_node.pose)
        dx = dy = 0
        if next_pose[0] - near_pose[0] > 0: dx = 1
        elif next_pose[0] - near_pose[0] < 0: dx = -1
        if next_pose[1] - near_pose[1] > 0: dy = 1
        elif next_pose[1] - near_pose[1] < 0: dy = -1

        success,new_pose = self.update_when_collision(near_pose,dx,dy)

        return success,new_pose,near_node


    def update_when_collision(self,near_pose,dx,dy):
        if dx|dy == 0:
            return False,near_pose

        new_pose = near_pose
        success = False
        flag0 = self.block_list[near_pose[0] + dx][near_pose[1] + dy]
        if dx&dy == 0:
            if flag0 == 0:
                new_pose[0] += dx
                new_pose[1] += dy
                success = True
        else:
            flag1 = self.block_list[near_pose[0]+dx][near_pose[1]]
            flag2 = self.block_list[near_pose[0]][near_pose[1]+dy]
            if  flag0 == 1:
                if flag1 == 0:
                    new_pose[0] += dx
                    success = True
                elif flag2 == 0:
                    new_pose[1] += dy
                    success = True
            else:
                if flag1&flag2 == 0:
                    new_pose[0] += dx
                    new_pose[1] += dy
                    success = True

        return success,new_pose

    def add_block(self,block_pose):
        self.block_list = block_pose

        # 将起点终点都强制设为可行区域
        if  self.block_list[self.root.pose[0]][self.root.pose[1]] == 1:
            self.block_list[self.root.pose[0]][self.root.pose[1]] = 0
        if self.block_list[self.end_pose[0]][self.end_pose[1]] == 1:
            self.block_list[self.end_pose[0]][self.end_pose[1]] = 0

        # 画出障碍物地图
        for i in range(block_pose.shape[0]):
            for j in range(block_pose.shape[1]):
                if (block_pose[i][j] == 1):
                    pygame.draw.rect(self.window, block_color, (j * self.grid, i * self.grid, self.grid, self.grid))
                else:
                    pygame.draw.rect(self.window, free_color, (j * self.grid, i * self.grid, self.grid, self.grid))

        # 画出起点和终点
        pygame.draw.rect(self.window, begin_color,(self.root.pose[1] * self.grid, self.root.pose[0] * self.grid, self.grid, self.grid))
        pygame.draw.rect(self.window, goal_color,(self.end_pose[1] * self.grid, self.end_pose[0] * self.grid, self.grid, self.grid))


if __name__ == '__main__':
    map_size = [20, 20]
    rrt = RRT(map_size[0], map_size[1], np.array([0, 0],dtype=int), np.array([15, 10],dtype=int),epsilon=0.3)  # 18,6
    # block_pose = np.random.randint(0, 2, size=map_size, dtype=int)
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
    rrt.add_block(block_pose)
    rrt.run()
