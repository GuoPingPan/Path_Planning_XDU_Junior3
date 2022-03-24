'''
    RRT_STAR(Rapidly-exploring Random Trees Star)
    随机快速扩展树*

    author: guopingpan
    email: 731061720@qq.com
            or panguoping02@gmail.com

    version: v2.0

    This project is using rrt—star algorithm to deal with path planning jobs.
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
        pose(np.array): 节点坐标位置
        father(Node):   父节点
        cost(float):    节点代价

    Functions:
        update_father:  添加父节点，更新代价

    '''
    def __init__(self,pose):
        self.pose = pose
        self.father = None
        self.cost = 0
        self.childs = []

    def update_father(self,father):
        self.father = father
        self.cost = father.cost + np.linalg.norm(self.pose-father.pose)

    def add_child(self,node):
        self.childs.append(node)

    def remove_child(self,node):
        self.childs.remove(node)

class RRT_STAR:
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

        radius(float):     更新节点的半径

    Tips:
        radius的作用是：搜索与 new_node 距离<radius的点，取代价最小者为父节点
                      同时将<radius的点，计算以 new_node 为父节点的代价，若小于原来的代价，则更新其父节点为 new_pose

    Functions:
        run: 运行主程序
        get_path: 回溯获得路径

        get_new_pose_and_nearest_index: 获得 new_pose 和 最近点的索引 nearest_index
            - update_when_collision:    碰撞检测，在 get_new_pose_and_nearest_index 中调用
                                        在发生碰撞时尝试调整并获得合法的 new_pose

        search_new_node_father: 创建 new_node,在<radius的节点中寻找 cost 最小者作为 new_node 的父节点
        rewire_near_node: 计算<radius的节点以 new_node 作为父节点的代价，并与原来的代价比较
                          若变小，则更新其父节点为 new_pose，否则不做调整

        quit+display: 保持显示，直至收到退出指令
        add_block: 添加障碍物地图

    '''
    def __init__(self,X,Y,start,end,radius = 2):
        self.map_size = (X,Y)
        self.root = Node(pose=start)
        self.closed_list = [self.root]
        self.end_pose = end
        self.block_list = None
        self.end_node = None
        self.radius = radius
        self.epsilon = 0.3
        self.path = []

        self.out = False

        # display the process
        pygame.init()
        self.window = pygame.display.set_mode((640, 640))
        self.grid = int(640 / X)
        pygame.display.set_caption('RRT_STRA搜寻路径')
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

                # 搜索下一个新点位置new_pose，距离next_pose最近点的索引nearest_index
                success,new_pose,nearest_index = self.get_new_pose_and_nearest_index(next_pose)

                # 成功搜索到new_pose
                if success:
                    # 搜索cost最小者作为new_pose的父亲
                    new_node, near_node_list = self.search_new_node_father(new_pose)

                    # 搜索到end_pose，成功找到路径
                    if np.linalg.norm(new_pose-self.end_pose) == 0:
                        self.end_node = new_node
                        self.closed_list.append(self.end_node)
                        # break
                    # 搜索成功，生成树枝
                    else:
                        self.rewire_near_node(near_node_list, new_node)
                        self.closed_list.append(new_node)

                        pygame.draw.rect(self.window, node_color, (new_pose[1] * self.grid, new_pose[0] * self.grid, self.grid, self.grid))

                self.clock.tick(fps) # 这里频率可调，来显示刷新快慢
                pygame.display.flip()


            self.quit()

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
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    position = self.get_position(event.pos)
                    pygame.draw.rect(self.window, new_block_color,(position[1] * self.grid, position[0] * self.grid, self.grid, self.grid))
                    pygame.display.flip()
                    if quit:
                        if self.need_modify(position):
                            pygame.display.flip()
                            quit =False

    def display(self):
        for pose in self.path:
            pygame.draw.rect(self.window, path_color,(pose[1] * self.grid, pose[0] * self.grid, self.grid, self.grid))

    def get_new_pose_and_nearest_index(self,next_pose):

        # 获得最近点的索引
        nearest_index = np.argmin([np.linalg.norm(node.pose - next_pose) for node in self.closed_list])
        nearest_pose = np.copy(self.closed_list[nearest_index].pose)

        # 搜索new_pose
        dx = dy = 0
        if next_pose[0] - nearest_pose[0] > 0: dx = 1
        elif next_pose[0] - nearest_pose[0] < 0: dx = -1
        if next_pose[1] - nearest_pose[1] > 0: dy = 1
        elif next_pose[1] - nearest_pose[1] < 0: dy = -1
        success,new_pose = self.update_when_collision(nearest_pose,dx,dy)

        return success,new_pose,nearest_index

    def update_when_collision(self,near_pose,dx,dy):
        if dx|dy == 0:
            return False,near_pose

        new_pose = near_pose
        success = False
        flag0 = self.block_list[near_pose[0] + dx][near_pose[1] + dy]
        # 不是对角线
        if dx&dy == 0:
            # 可走
            if flag0 == 0:
                new_pose[0] += dx
                new_pose[1] += dy
                success = True
        # 对角线
        else:
            # 对角线两边的两个节点
            flag1 = self.block_list[near_pose[0]+dx][near_pose[1]]
            flag2 = self.block_list[near_pose[0]][near_pose[1]+dy]
            # 对角线不可以走
            if  flag0 == 1:
                # 两边随便选一个点走
                if flag1 == 0:
                    new_pose[0] += dx
                    success = True
                elif flag2 == 0:
                    new_pose[1] += dy
                    success = True
            # 对角线可以走
            else:
                # 两边没有都被挡
                if flag1&flag2 == 0:
                    new_pose[0] += dx
                    new_pose[1] += dy
                    success = True

        return success,new_pose

    def search_new_node_father(self,new_pose):

        # 默认new_node的father为最近点
        new_node = Node(pose=new_pose)
        self.block_list[new_pose[0]][new_pose[1]] = 1
        # new_node.update_father(self.closed_list[nearest_index])

        near_node_list = []
        for node in self.closed_list:
            dist = np.linalg.norm(node.pose - new_pose)
            if dist < self.radius:
                near_node_list.append(node)

        # 以cost小的点更新new_node的father
        cost_less_index = np.argmin([node.cost+np.linalg.norm(node.pose-new_node.pose)] for node in near_node_list)
        new_node.update_father(near_node_list[cost_less_index])
        near_node_list[cost_less_index].add_child(new_node)
        near_node_list.remove(near_node_list[cost_less_index])
        return new_node,near_node_list

    def rewire_near_node(self,near_node_list,new_node):
        # 尝试更新邻近点的父节点，以降低cost
        for node in near_node_list:
            if node.cost > new_node.cost + np.linalg.norm(node.pose-new_node.pose):
                node_father = node.father
                node_father.remove_child(node)
                node.update_father(new_node)
                new_node.add_child(node)

    def add_block(self,block_pose):
        self.block_list = block_pose

        # 将起点终点都强制设为可行区域
        if  self.block_list[self.root.pose[0]][self.root.pose[1]] == 1:
            self.block_list[self.root.pose[0]][self.root.pose[1]] = 0
        if self.block_list[self.end_pose[0]][self.end_pose[1]] == 1:
            self.block_list[self.end_pose[0]][self.end_pose[1]] = 1

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

    def need_modify(self,position):
        self.block_list[position[0]][position[1]] = 1
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
            node = cut.pop(0)
            node_father = node.father
            node_father.remove_child(node)

            for node in cut:
                self.closed_list.remove(node)
                self.block_list[node.pose[0]][node.pose[1]] = 0
                pygame.draw.rect(self.window, free_color,(node.pose[1] * self.grid, node.pose[0] * self.grid, self.grid, self.grid))

            pygame.draw.rect(self.window, goal_color,(self.end_pose[1] * self.grid, self.end_pose[0] * self.grid, self.grid, self.grid))

            self.end_node = None
            return True

        return False

    def get_mouse_postion(self,pose):
        return np.array([int(pose[1]/self.grid),int(pose[0]/self.grid)])

    def get_position(self,pos):
        position=[0,0]
        for i in range(20):
            if pos[0]>=i*32 and pos[0]<(i+1)*32:
                position[1]=i
            if pos[1] >= i * 32 and pos[1] < (i + 1) * 32:
                position[0] = i
        return position


if __name__ == '__main__':
    map_size = [20, 20]
    rrt = RRT_STAR(map_size[0], map_size[1], np.array([0, 0],dtype=int), np.array([17, 9],dtype=int))  # 18,6
    # block_pose = np.random.randint(0, 2, size=map_size, dtype=int)
    block_pose = np.array([[0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
                           [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
                           [0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
                           [0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0],
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
    rrt.quit()