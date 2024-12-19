"""DQN Synchronous Train Torch version"""
from threading import Thread
import os
import glob # 用于添加carla.egg。环境中装有.whl可忽略
import sys
import random
import time
from collections import deque # 双端队列

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm # 在Python长循环中添加一个进度提示信息，用户只需要封装任意的迭代器tqdm(iterator)

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

from CarEnv_studio import CarEnv, IM_WIDTH, IM_HEIGHT,camera_queue1, camera_queue2 

from key import on_key_pressed
import keyboard

"""
    Policynet():
    input: state([batch, 7,  height, width])
    return: action([batch, 2])
"""
class Policynet(nn.Module):
    def __init__(self, IM_HEIGHT, IM_WIDTH):
        super(Policynet, self).__init__() 
        # images的卷积层+全连接层
        self.conv1 = nn.Sequential(
            # nn.BatchNorm2d(7),
            nn.Conv2d(6, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.BatchNorm2d(64),
        )
        self.conv_fc1 = nn.Sequential(
            nn.Linear(int(64 * (IM_HEIGHT/8) * (IM_WIDTH/8)), 64),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )
        self.conv_fc2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )

        # attributes全连接层
        self.bn1 = nn.BatchNorm1d(20)
        self.fc1 = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )
        
        # 拼接后的全连接层 32 + 32 = 64 --> 32 -->16 -->2
        self.cat_fc1 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )
        self.cat_fc2 = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )
        self.cat_fc3 = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )
        self.cat_fc4 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )
        self.cat_fc5 = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )
        self.cat_fc6 = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(16, 2), 
            nn.Tanh()
        )

    def forward(self, images, attributes):
        conv1_out = self.conv1(images) 
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv3_res = conv3_out.reshape(conv3_out.size(0), -1) # --> (76800)
        conv_fc1_out = self.conv_fc1(conv3_res) # 76800 --> (64)
        conv_fc2_out = self.conv_fc2(conv_fc1_out) # (32)
        # print("conv_fc2_out", conv_fc2_out.shape) #torch.Size([64, 32])

        attributes = self.bn1(attributes) # 将20个特征先批归一化
        fc1_out = self.fc1(attributes) # 20 --> 64
        fc2_out = self.fc2(fc1_out) # 64 --> 32
        fc3_out = self.fc3(fc2_out) # 32 --> 32
        # print("fc3_out", fc3_out.shape) # torch.Size([64, 32])

        cat = torch.cat(( conv_fc2_out, fc3_out), 1) # 32 + 32 = 64 --> 32
        # print("cat", cat.shape) # torch.Size([64, 64])
        cat_fc1_out = self.cat_fc1(cat) # 32 --> 16
        cat_fc2_out = self.cat_fc2(cat_fc1_out) # 16 --> 2 
        cat_fc3_out = self.cat_fc3(cat_fc2_out) # 
        cat_fc4_out = self.cat_fc4(cat_fc3_out) # 
        cat_fc5_out = self.cat_fc5(cat_fc4_out) # 
        cat_fc6_out = self.cat_fc6(cat_fc5_out) # 

        return cat_fc6_out # (batch, 2): 单一获取动作时会获得[(1, 2)]

"""
    QValueNet():
    input: state([batch, 7,  height, width]), action([batch, 2])]
    return: QValue(1)
"""
class QValueNet(nn.Module):
    def __init__(self, IM_HEIGHT, IM_WIDTH):
        super(QValueNet, self).__init__() 

        self.IM_WIDTH = IM_WIDTH
        self.IM_HEIGHT = IM_HEIGHT

        self.conv1 = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.dense = nn.Sequential(
            nn.Linear(int(64 * (IM_HEIGHT/8) * (IM_WIDTH/8)), 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1), # action(batch, 2)
        )


    def forward(self, x, action): #拼接state和action，其中action([batch, 2])，需要拼接为两层
        tensor_list = []
        for i in range(MINIBATCH_SIZE): #x: (b,3,h,w)->(b,5,h,w)
            action = action.squeeze(1)
            a0 = action[i][0] #油刹
            a1 = action[i][1] #方向
            a0_array = torch.ones((1, self.IM_HEIGHT, self.IM_WIDTH)).to(device)*a0 # （ 1, h, w）
            a1_array = torch.ones((1, self.IM_HEIGHT, self.IM_WIDTH)).to(device)*a1 # （ 1, h, w）
            tensor_list.append(torch.cat([x[i], a0_array, a1_array], dim=0)) # [（ 9, h, w）]
        x = torch.stack(tensor_list) # x.shape torch.Size([2, 9, 240, 320]) batch==2时
        
        conv1_out = self.conv1(x) 
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.reshape(conv3_out.size(0), -1)
        out = self.dense(res)
        # print(out)
        return out # (batch, 2)

if torch.cuda.device_count() > 1:
    device = torch.device("cuda:0") 
else:
    device = torch.device("cuda:0")

class DDPG:
    def __init__(self,lr_actor=1e-4, lr_critic=1e-4):

        self.actor = Policynet(IM_HEIGHT, IM_WIDTH).to(device)
        self.critic = QValueNet(IM_HEIGHT, IM_WIDTH).to(device)
        self.target_actor = Policynet(IM_HEIGHT, IM_WIDTH).to(device)
        self.target_critic = QValueNet(IM_HEIGHT, IM_WIDTH).to(device)

        #初始化同步目标网络
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        #只更新主网络的参数
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor) 
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic) 

        #启动预训练模型
        if TRAINED_MODEL:
            if os.path.exists(trained_model_dir):
                checkpoint = torch.load(trained_model_dir)
                self.actor.load_state_dict(checkpoint['model'])
                self.actor_optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint['epoch']
                print('加载 epoch {} 成功！'.format(start_epoch))   

        self.sigma = 0.1 #高斯噪声标准差
        self.tau = 0.005 #软更新参数
        self.gamma = 0.99 #折扣因子
        
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE) #上限5000的经验回放池队列

        self.terminate = False #没结束循环训练，当全部游戏次数跑完后此处会改为True，停止进程中的采样训练
        self.quit = False # ubuntu中不得使用
    
    def get_action(self, images, attributes): # state(16,c,h,w)
        self.actor.eval() # 提取动作时无需反向传播，关闭梯度节省显存。！！！有个问题，双线程时关闭梯度会影响线程2的模型反向传播训练
        with torch.no_grad():
            images = torch.tensor(images, dtype=torch.float).to(device) # (hwc)
            images = images/255 
            images = images.permute(2, 0, 1) # (chw)
            images = images.unsqueeze(dim=0) # (n, h, w, c) 

            action = self.actor(images, attributes)# action(batch, 2)
            
            #给动作添加噪声，增加探索
            action = action.clone() + torch.tensor(self.sigma * np.random.randn(2)).to(device) #self.action_dim = 2
            
        self.actor.train() # 获取动作后开启梯度计算
        # print(action)
        return action # action(batch, 2)

    #将replay_memory、update_replay_memory()均现在class DQNAgent内部，无需从类外再传入经验回放池队列变量
    def update_replay_memory(self, transition): 
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(self.tau * param.data + (1.0 - self.tau) * param_target.data)

    def train(self):
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE) #每次采样16个轨迹作为一批次同时训练
        # transition: (current_state, data, new_current_state, new_data, action, act_expert)
        # current_state: (240, 320, 6)
        # new_current_state: (240, 320, 6) 
        # data = [*location, *start_point, *destination, *forward_vector, velocity, *acceleration, *angular_velocity, reward]: tensor([1, 20])

        current_states = []
        attributes = []
        new_current_states = []
        new_attributes = []
        actions = []
        actions_experts = []
        rewards = []

        for transition in minibatch:
            
            current_states.append(transition[0])
            attributes.append(transition[1].tolist()) 
            # print(transition[1].shape)

            new_current_states.append(transition[2])
            new_attributes.append(transition[3].tolist()) 

            actions.append(transition[4].tolist())
            actions_experts.append(transition[5].tolist())

            rewards.append(transition[1][0][-1])

        # 16个随机样本的current_states，16x(h, w, c) --> (16, c, h, w)
        current_states = np.array(current_states)/255 #(16, h, w, c)
        current_states = torch.from_numpy(current_states)  #Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
        current_states = current_states.to(torch.float32).to(device)
        current_states = current_states.permute(0,3,1,2) # (16, c, h, w)

        # 16个随机样本的new_current_states
        new_current_states = np.array(new_current_states)/255
        new_current_states = torch.from_numpy(new_current_states) 
        new_current_states = new_current_states.to(torch.float32).to(device)
        new_current_states = new_current_states.permute(0,3,1,2) # (16, c, h, w)

        attributes = torch.tensor(attributes).to(torch.float32).to(device)
        attributes = attributes.squeeze(1) # 直接去除为1的维度

        new_attributes = torch.tensor(new_attributes).to(torch.float32).to(device)
        new_attributes = new_attributes.squeeze(1) # 直接去除为1的维度

        rewards = torch.tensor(rewards).to(torch.float32).to(device) # (batch)
        rewards = rewards.reshape(-1, 1) #(batch)-->(batch,1)

        actions = torch.tensor(actions).to(torch.float32).to(device)
        actions_experts = torch.tensor(actions_experts).to(torch.float32).to(device)

        future_qs_list = self.target_critic(new_current_states, self.target_actor(new_current_states, new_attributes)) #  Qw-(s', Π-（a'）)=>(batch, 1) target全用target网络计算，再用真网络计算的Q去拟合。
        target_qs_list = rewards + self.gamma * future_qs_list

        critic_loss = torch.mean(nn.functional.mse_loss(self.critic(current_states, actions), target_qs_list)).to(device) 

        self.critic_optimizer.zero_grad() # 梯度清零
        critic_loss.backward() # 产生梯度 同时对critic和actor产生梯度！其实不影响。只更actor前会清零.不应该改变actor梯度！！！！
        self.critic_optimizer.step() # 根据梯度更新
        
        actor_act = self.actor(current_states, attributes)
        actor_loss = -torch.mean(self.critic(current_states, actor_act)).to(device) #Q(s, Π（s）)上升

        self.actor_optimizer.zero_grad() # 梯度清零
        actor_loss.backward() # 产生梯度 同时对critic和actor产生梯度！其实不影响。只更critic前会清零
        self.actor_optimizer.step() # 使用target_critic做loss后，actor和critic更新的先后顺序也无所谓了

        self.soft_update(self.critic, self.target_critic) # 目前是每次训练都更新。需适当降频
        self.soft_update(self.actor, self.target_actor)

        return critic_loss.item(), actor_loss.item()
    
    def train_in_loop(self):
        total_train_step = 0
        while True:
            if self.terminate: #当所有回合批样本收集跑完后此处改为true，只要没跑够，此处会一直支线程进行训练
                return
            
            if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
                print(len(self.replay_memory))
                # return

            else: # 当经验回放池样本数量大于MIN_REPLAY_MEMORY_SIZE才会开始采样训练
                critic_loss, actor_loss = self.train()
                print("total_train_step", total_train_step, "critic_loss:", critic_loss, "actor_loss:", actor_loss)
                total_train_step += 1

                if LOG:
                    writer.add_scalar("critic_loss", critic_loss, total_train_step)
                    writer.add_scalar("actor_loss", actor_loss, total_train_step)
                
                if total_train_step % 1000 == 0:
                    if SAVE:
                        state = {'model':agent.actor.state_dict(), 'optimizer':agent.actor_optimizer.state_dict(), 'epoch': episode}
                        torch.save(state, path)

    def esc_quit(self):
        # 按esc键时，关闭键盘监听线程
        keyboard.on_press_key('esc', on_key_pressed)
        keyboard.wait('esc')  # 等待按下esc键后停止监听 时机==按下 》》释放
        self.quit = True


def camera_deal(camera_queue):
    new_state1 = camera_queue.get()
    i_1 = np.array(new_state1.raw_data) # .raw_data：Array of BGRA 32-bit pixels
    #print(i.shape)
    i2_1 = i_1.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3_1 = i2_1[:, :, :3] # （h, w, 3）= (480, 640, 3)
    if SHOW_PREVIEW:
        cv2.imshow("i3_1", i3_1)
        cv2.waitKey(1)
    return i3_1


if __name__ == '__main__':

    now = time.ctime(time.time())
    now = now.replace(" ","_").replace(":", "_")

    SHOW_PREVIEW = False # 训练时播放摄像镜头
    LOG = True # 训练时向tensorboard中写入记录
    SAVE = True # 保存模型
    TRAINED_MODEL = True # 是否有预训练模型
    expert_act = False #是否开启专家演示
    com_pi = False# 开启混合策略

    if SAVE:
        if not os.path.isdir('./DDPG_model'):
            os.makedirs('./DDPG_model')
        path='./DDPG_model/model_{}.pth'.format(now)

    if LOG:
        if not os.path.isdir('./DDPG_logs/{}'.format(now)):
            os.makedirs('./DDPG_logs/{}'.format(now))
        writer = SummaryWriter('./DDPG_logs/{}'.format(now))

    trained_model_dir = r"Dagger_model/model_Fri_Sep__6_21_17_38_2024.pth" # shotgundagger2
    # trained_model_dir = r"Dagger_model/model_Thu_Aug_15_01_18_02_2024.pth" # 混合衰减dagger（效果不好）

    beta = 0.9999 # 混合策略衰减系数
    REPLAY_MEMORY_SIZE = 1000 # 经验回放池最大容量
    MIN_REPLAY_MEMORY_SIZE = 1000# 抽样训练开始时经验回放池的最小样本数
    MINIBATCH_SIZE = 16 # 每次从经验回放池的采样数（作为训练的同一批次）   此大小影响运算速度/显存
    # UPDATE_TARGET_EVERY = 5 # 同步target网络的训练次数 !!!!!!!!!!!!!!!!!!!!!!!!此处无用！！！！！！！！！！！！！！！！！！！
    EPISODES = 1000 # 游戏进行总次数
    DISCOUNT = 0.99 # 贝尔曼公式中折扣因子γ
    lr_actor = 1e-4 # from dagger best
    lr_critic = 1e-4

    env = CarEnv()    
    agent = DDPG(lr_actor, lr_critic)

    Original_settings = env.original_settings # 将原设置传出来保存

    # 训练线程
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True) #创建一个线程，调用的函数是train_in_loop
    trainer_thread.start() #此处会直接往下走，同时线程分支（train_in_loop，训练一批）开始运行

    # 退出线程
    quit_thread = Thread(target=agent.esc_quit, daemon=True) #创建一个线程，调用的函数是train_in_loop
    quit_thread.start()

    episode_num = 0 # 游戏进行的次数

    all_average_reward = 0

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'): # 1~1000 EPISODE
        env.collision_hist = [] # 记录碰撞发生的列表
        episode_reward = 0 # 每次游戏所有step累计奖励

        # Reset environment and get initial state
        env.reset() #此处reset里会先tick()一下，往队列里传入初始图像

        #以下为初始帧的s,a,r,done,kmh,dis
        current_state1 = camera_deal(camera_queue1)
        current_state2 = camera_deal(camera_queue2)
        current_state = np.concatenate((current_state1, current_state2), axis=2) # （h, w, 3）= (480, 640, 3+3+1 = 7)

        action = torch.tensor([[0, 0]]).to(device)  # torch.Size([1, 2])

        # data = [*location, *start_point, *destination, *forward_vector, velocity, *acceleration, *angular_velocity, reward]
        data = [0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0] # (20,)
        data = torch.tensor([data]).to(device) 

        dist_to_start = 0.0
        dist_to_end = env.dist_to_end_init

        episode_start = time.time()
        episode_num += 1
        episode_steps = 1

        if agent.quit:
            print("Esc break1")
            break

        # Play for given number of seconds only
        while True:

            #synchronous
            env.world.tick()

            state1 = camera_deal(camera_queue1)
            state1 = camera_deal(camera_queue2)

            new_current_state = np.concatenate((state1, state1), axis=2) # （h, w, 3+3+1）= (480, 640, 7)

            # reward, done, _ = env.step(action)
            done, new_data, act_expert, new_dist_to_start, new_dist_to_end = env.step(action, episode_steps, dist_to_start, dist_to_end )


            new_data = [new_data] # data预留batch_size维度 (1, 20)
            new_data = torch.tensor(new_data).cuda().float() # tensor([1, 20])
            # data = [*location, *start_point, *destination, *forward_vector, velocity, *acceleration, *angular_velocity, reward]

            # # 增加经验池样本丰富度,训练前随机采样
            # if len(agent.replay_memory) >= REPLAY_MEMORY_SIZE:
            #     action = agent.get_action(new_current_state, new_data)
            # else:
            #     action = torch.tensor([[torch.rand(1)*2 - 1, torch.rand(1)*2 - 1]]).to(device)
            #     print(len(agent.replay_memory))

            action = agent.get_action(new_current_state, new_data)

            # transition: (current_state, data, new_current_state, new_data, action, act_expert, reward)
            agent.update_replay_memory((current_state, data, new_current_state, new_data, action, act_expert)) 

            current_state = new_current_state.copy() #array直接复制会浅拷贝共用内存，此处需深拷贝保持二者独立性 (480, 640, 3)
            data = new_data
            dist_to_start = new_dist_to_start
            dist_to_end = new_dist_to_end

            episode_reward += data[0][-1]

            # set the sectator to follow the ego vehicle
            spectator = env.world.get_spectator()
            transform = env.vehicle.get_transform()
            spectator.set_transform(carla.Transform(transform.location + carla.Location(z=20),
                                                carla.Rotation(pitch=-90)))
            
            episode_steps += 1

            if done:
                agent.update_replay_memory((current_state, data, new_current_state, new_data, action, act_expert))
                break

            if agent.quit:
                print("Esc break2")
                break

        # End of episode - destroy agents
        for actor in env.actor_list:
            actor.destroy()

        episode_average_reward = episode_reward/episode_steps

        # 记录每一次游戏的平均奖励
        if LOG:
            writer.add_scalar("average_reward", episode_average_reward, episode_num)

        all_average_reward += episode_average_reward

        # 每100个episode保存一次模型，输出一下近100次的平均奖励
        if episode_num % 100 == 0:
            if SAVE:
                    state = {'actor_model':agent.actor.state_dict(), 'optimizer':agent.actor_optimizer.state_dict(), 'epoch': episode}
                    torch.save(state, path)
                    state = {'critic_model':agent.critic.state_dict(), 'optimizer':agent.critic_optimizer.state_dict(), 'epoch': episode}
                    torch.save(state, path)

            print("all_average_reward", all_average_reward/100)
            all_average_reward = 0

    #num_episodes(1000次)跑完了
    agent.terminate = True
    trainer_thread.join()#将调用join的线程优先执行，当前正在执行的线程阻塞，直到调用join方法的线程执行完毕或者被打断，主要用于线程之间的交互。

    if SAVE:
        state = {'model':agent.actor.state_dict(), 'optimizer':agent.actor_optimizer.state_dict(), 'epoch': episode}
        torch.save(state, path)


    env.world.apply_settings(Original_settings)
