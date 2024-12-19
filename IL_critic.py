import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import os

def get_states_actions(data_index ,batch_size): # 采集的数据中第四个alpha通道已剔除
    # 采样
    minibatch = random.sample(data_index, batch_size)

    imgs = []
    actions = []
    attributes = []

    new_imgs = []
    new_actions = []
    new_attributes = []

    for i in minibatch:
        camera1_img = mpimg.imread('./1w_300step_4fps_attributes/camera1//{}.jpg'.format(i))
        camera2_img = mpimg.imread('./1w_300step_4fps_attributes/camera2//{}.jpg'.format(i))
        img = np.concatenate((camera1_img, camera2_img), axis=2) # 将影像叠置拼接
        imgs.append(img)

        new_camera1_img = mpimg.imread('./1w_300step_4fps_attributes/camera1//{}.jpg'.format(i+1))
        new_camera2_img = mpimg.imread('./1w_300step_4fps_attributes/camera2//{}.jpg'.format(i+1))
        new_img = np.concatenate((new_camera1_img, new_camera2_img), axis=2) # 将影像叠置拼接
        new_imgs.append(new_img)

        location = load_dict["{}".format(i)][1] # (3)
        start_point = load_dict["{}".format(i)][2] # (3)
        destination = load_dict["{}".format(i)][3] # (3)
        forward_vector = load_dict["{}".format(i)][4] # (3)
        velocity = load_dict["{}".format(i)][5] # (1)
        acceleration = load_dict["{}".format(i)][6] # (3)
        angular_velocity = load_dict["{}".format(i)][7] # (3)
        reward = load_dict["{}".format(i)][8] # (1)

        new_location = load_dict["{}".format(i+1)][1] # (3)
        new_start_point = load_dict["{}".format(i+1)][2] # (3)
        new_destination = load_dict["{}".format(i+1)][3] # (3)
        new_forward_vector = load_dict["{}".format(i+1)][4] # (3)
        new_velocity = load_dict["{}".format(i+1)][5] # (1)
        new_acceleration = load_dict["{}".format(i+1)][6] # (3)
        new_angular_velocity = load_dict["{}".format(i+1)][7] # (3)
        new_reward = load_dict["{}".format(i+1)][8] # (1)

        attribute = [*location, *start_point, *destination, *forward_vector, velocity, *acceleration, *angular_velocity, reward]
        attributes.append(attribute) # len(20)

        new_attribute = [*new_location, *new_start_point, *new_destination, *new_forward_vector, new_velocity, *new_acceleration, *new_angular_velocity, new_reward]
        new_attributes.append(new_attribute) # len(20)

        # print(attribute)

        action = load_dict["{}".format(i)][0]
        actions.append(action)

        new_action = load_dict["{}".format(i+1)][0]
        new_actions.append(new_action)

    imgs = np.array(imgs)/255
    imgs = torch.from_numpy(imgs) 
    imgs = imgs.to(torch.float32).to(device)
    imgs = imgs.permute(0,3,1,2) # (b, c, h, w) --> torch.Size([batch_size, 6, 240, 320])

    new_imgs = np.array(new_imgs)/255
    new_imgs = torch.from_numpy(new_imgs) 
    new_imgs = new_imgs.to(torch.float32).to(device)
    new_imgs = new_imgs.permute(0,3,1,2) # (b, c, h, w) --> torch.Size([batch_size, 6, 240, 320])

    attributes = np.array(attributes) # torch.Size([batch_size,20])
    attributes = torch.tensor(attributes).to(device).float()
    new_attributes = np.array(new_attributes) # torch.Size([batch_size,20])
    new_attributes = torch.tensor(new_attributes).to(device).float()
    

    actions = np.array(actions)
    actions = torch.tensor(actions).to(device).float()
    new_actions = np.array(new_actions)
    new_actions = torch.tensor(new_actions).to(device).float()

    return imgs, new_imgs, attributes, new_attributes, actions, new_actions

# data = [[throttle, steer], location, start_point, destination, forward_vector, velocity, acceleration, angular_velocity, reward]

def soft_update(net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(tau * param.data + (1.0 - tau) * param_target.data)

class Policynet_cat_fc_pro(nn.Module):
    def __init__(self, IM_HEIGHT, IM_WIDTH):
        super(Policynet_cat_fc_pro, self).__init__() 
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

        return cat_fc6_out # (batch, 2)

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
            nn.Linear(64, 1), # action(batch, 1)
        )


    def forward(self, x, action): #拼接state和action，其中action([batch, 2])，需要拼接为两层
        tensor_list = []
        for i in range(batch_size): #x: (b,3,h,w)->(b,5,h,w)
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
        return out # (batch, 1)
    

if __name__ == "__main__":

    now = time.ctime(time.time())
    now = now.replace(" ","_").replace(":", "_")

    device = torch.device("cuda:0")

    epoch = 1000
    batch_size = 64 # 128时dagger policy + 垃圾critic会爆显存
    lr_critic = 1e-4
    gamma = 0.99 #折扣因子
    tau = 0.005 #软更新参数
    
    SAVE = True
    LOG = True


    # 专家样本数据集
    load_dict = np.load('./1w_300step_4fps_attributes/record_dic.npy', allow_pickle=True).item()
    
    # 划分数据集
    partion_line = len(load_dict)//4 * 3

    list_dic_index = range(len(load_dict))

    tarin_data_index = set(random.sample(list_dic_index, partion_line))
    test_data_index = set(list_dic_index) - tarin_data_index
    # 涉及未来状态和当前状态为一对，因此样本不要最后一个
    tarin_data_index = list(tarin_data_index)
    test_data_index = list(test_data_index)
    tarin_data_index.pop()
    test_data_index.pop()

    # 预训练目标策略网络模型
    trained_model_dir = r"Dagger_model/model_Fri_Sep__6_21_17_38_2024.pth" # Shotgundagger2

    # 目标策略网络（不训练）以及价值网络（待训练，含优化器及目标价值网络）
    target_actor = Policynet_cat_fc_pro(240, 320)
    target_actor = target_actor.to(device)

    critic = QValueNet(240, 320).to(device)

    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr_critic) 

    target_critic = QValueNet(240, 320).to(device)
    target_critic.load_state_dict(critic.state_dict())

    # 加载预训练目标策略网络模型
    if os.path.exists(trained_model_dir):
        checkpoint = torch.load(trained_model_dir)
        target_actor.load_state_dict(checkpoint['model'])
        # actor_optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('加载 epoch {} 成功！'.format(start_epoch))   

    if SAVE:
        if not os.path.isdir('./critic_model'):
            os.makedirs('./critic_model')
        path='./critic_model/model_{}.pth'.format(now)

    if LOG:
        if not os.path.isdir('./critic_logs/{}'.format(now)):
            os.makedirs('./critic_logs/{}'.format(now))
        writer = SummaryWriter('./critic_logs/{}'.format(now))


    total_train_step = 0
    total_test_step = 0

    for i in range(epoch):
        target_actor.train() # 开启训练模式
        print("------------------第{}轮训练开始-----------------".format(i+1))

        epoch_add_loss = 0
        epoch_step = 0

        for j in range(len(tarin_data_index)//batch_size): # 每轮把所有数据的量至少抽样过一遍

            imgs, new_imgs, attributes, new_attributes, actions, new_actions = get_states_actions(tarin_data_index ,batch_size)
            rewards = attributes[:, -1].unsqueeze(1) # tensor[:, -1] 提取张量中每行最后一个元素，得到形状为 [32] 的一维张量。unsqueeze(1) 在维度 1 添加一个维度，使结果形状变为 [32, 1]。

            target_action = target_actor(new_imgs, new_attributes)

            future_qs_list = target_critic(new_imgs, target_action) #  Qw-(s', Π-（a'）)=>(batch, 1) target全用target网络计算，再用真网络计算的Q去拟合。

            target_qs_list = rewards + gamma * future_qs_list

            critic_loss = torch.mean(nn.functional.mse_loss(critic(imgs, actions), target_qs_list)).to(device) 
            # Using a target size (torch.Size([32, 20])) that is different to the input size (torch.Size([32, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.??????????????????????????看结果没练出来

            critic_optimizer.zero_grad() # 梯度清零
            critic_loss.backward() # 产生梯度 同时对critic和actor产生梯度！其实不影响。只更actor前会清零.不应该改变actor梯度！！！！
            critic_optimizer.step() # 根据梯度更新

            # 目前是每次训练都软更新。需适当降频
            soft_update(critic, target_critic)

            total_train_step += 1
            epoch_add_loss += critic_loss
            epoch_step += 1

            if total_train_step % 100 == 0:
                print("第{}轮，第{}步，loss：{}".format(i+1, total_train_step, critic_loss.item())) #不加item输出的是tensor，加了输出的是数字

            if LOG:
                writer.add_scalar("critic_loss{}", critic_loss.item(), total_train_step)

        print("训练集上平均loss：{}".format(epoch_add_loss.item()/epoch_step))

        if SAVE:
            state = {'model':critic.state_dict(), 'optimizer':critic_optimizer.state_dict(), 'epoch':i}
            torch.save(state, path)

        # #测试开始
        # target_actor.eval() # 开启测试模式
        # total_test_loss = 0
        # with torch.no_grad():
        #     for j in range(100):
        #         imgs, attributes, actions = get_states_actions(test_data_index ,batch_size)
        #         imgs = imgs.to(device)
        #         attributes = torch.tensor(attributes).to(device).float()
        #         actions = torch.tensor(actions).to(device).float()
        #         outputs = target_actor(imgs, attributes)

        #         loss_throttle = loss_fn_throttle(outputs[:, 0], actions[:, 0])
        #         loss_steer = loss_fn_steer(outputs[:, 1], actions[:, 1])
        #         loss = loss_throttle + weight_loss_steer * loss_steer

        #         total_test_step += 1
        #         total_test_loss += loss
        #         # accuracy = (outputs.argmax(1) == labels).sum()
        #         # total_accuracy = total_accuracy + accuracy
        #         if LOG:
        #             writer.add_scalar("test_loss{}", loss.item(), total_test_step)

        #     print("测试集上平均loss：{}".format(total_test_loss.item()/100))
        #     # print("整体测试集上的准确率：{}".format(total_accuracy/test_data_size))
            
        #     # writer.add_scalar("test_accuracy",total_accuracy/test_data_size, total_test_step)

    writer.close()