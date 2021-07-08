import torch
import torch.nn as nn
import torch.optim
import config as c
import FrEIA.framework as Ff
import FrEIA.modules as Fm

class CondNet(nn.Module):
    '''conditioning network'''
    def __init__(self):
        super().__init__()

        class Flatten(nn.Module):
            def __init__(self, *args):
                super().__init__()
            def forward(self, x):
                return x.view(x.shape[0], -1)

        self.resolution_levels = nn.ModuleList([
                           nn.Sequential(nn.Conv2d(3,  64, 3, padding=1),
                                         nn.LeakyReLU(),
                                         nn.Conv2d(64, 64, 3, padding=1)),

                           nn.Sequential(nn.LeakyReLU(),
                                         nn.Conv2d(64,  128, 3, padding=1),
                                         nn.LeakyReLU(),
                                         nn.Conv2d(128, 128, 3, padding=1, stride=2)),

                           nn.Sequential(nn.LeakyReLU(),
                                         nn.Conv2d(128, 128, 3, padding=1, stride=2)),

                           nn.Sequential(nn.LeakyReLU(),
                                         nn.AvgPool2d(4),
                                         Flatten(),
                                         nn.Linear(2048, 512))])

    def forward(self, c):
        outputs = [c]
        for m in self.resolution_levels:
            x = m(outputs[-1])
            outputs.append(x)
        return outputs[1:]

class CINN(nn.Module):
    '''C-Glow, including the conditioning network'''
    def __init__(self, structure = False):
        super().__init__()

        self.structure = structure
        self.cinn = self.build_inn()
        self.cond_net = CondNet()

        self.trainable_parameters = [p for p in self.cinn.parameters() if p.requires_grad]
        for p in self.trainable_parameters:
            p.data = 0.02 * torch.randn_like(p)

        self.trainable_parameters += list(self.cond_net.parameters())
        self.num_trainable_parameters = self.calc_trainable_params()
        self.optimizer = torch.optim.Adam(self.trainable_parameters, lr=c.lr)

    def calc_trainable_params(self):
        num = 0
        for tp in self.trainable_parameters:
            sum = 0
            for i in range(len(tp.size())):
                if i == 0:
                    sum = tp.size()[i]
                else:
                    sum *= tp.size()[i]
            num += sum
        return num

    def build_inn(self):
        def sub_conv(ch_hidden, kernel):
            pad = kernel // 2
            return lambda ch_in, ch_out: nn.Sequential(
                                            nn.Conv2d(ch_in, ch_hidden, kernel, padding=pad),
                                            nn.ReLU(),
                                            nn.Conv2d(ch_hidden, ch_out, kernel, padding=pad))

        def sub_fc(ch_hidden):
            return lambda ch_in, ch_out: nn.Sequential(
                                            nn.Linear(ch_in, ch_hidden),
                                            nn.ReLU(),
                                            nn.Linear(ch_hidden, ch_out))

        nodes = [Ff.InputNode(c.channels, c.img_w, c.img_h)]
        # outputs of the cond. net at different resolution levels
        conditions = [Ff.ConditionNode(64, 64, 64),
                      Ff.ConditionNode(128, 32, 32),
                      Ff.ConditionNode(128, 16, 16),
                      Ff.ConditionNode(512)]

        split_nodes = []

        subnet = sub_conv(32, 3)
        for k in range(c.cluster_1):
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':1.0, 'subnet_args':{}},
                                 conditions=conditions[0], name=f"GLOW_1_{(k+1)}"))
        nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'rebalance':0.5}, name="Haar_1"))

        for k in range(c.cluster_2):
            subnet = sub_conv(64, 3 if k % 2 else 1)
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':1.0, 'subnet_args':{}},
                                 conditions=conditions[1], name=f"GLOW_2_{(k+1)}"))
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed':k}, name=f"PermuteRamdom_2_{(k+1)}"))
        #split off 6/8 ch
        nodes.append(Ff.Node(nodes[-1], Fm.Split, {'section_sizes':[3,9], 'dim':0}, name=f"Split_1"))
        split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {}, name="Flatten_1"))
        nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'rebalance':0.5}))

        for k in range(c.cluster_3):
            subnet = sub_conv(128, 3 if k % 2 else 1)
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':0.6, 'subnet_args':{}},
                                 conditions=conditions[2], name=f"GLOW_3_{(k+1)}"))
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed':k}, name=f"PermuteRamdom_3_{(k+1)}"))

        #split off 4/8 ch
        nodes.append(Ff.Node(nodes[-1], Fm.Split,
                             {'section_sizes':[6,6], 'dim':0}, name=f"Split_2"))
        split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {}))
        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='Flatten'))
        subnet = sub_fc(512)

        for k in range(c.cluster_4):
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':0.6, 'subnet_args':{}},
                                 conditions=conditions[3], name=f"GLOW_4_{(k+1)}"))
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed':k}, name=f"PermuteRamdom_4_{(k+1)}"))

        # concat everything
        nodes.append(Ff.Node([s.out0 for s in split_nodes] + [nodes[-1].out0],
                             Fm.Concat1d, {'dim':0}, name=f"Concat1d"))

        nodes.append(Ff.OutputNode(nodes[-1], name="Output"))

        return Ff.GraphINN(nodes + split_nodes + conditions, verbose=c.show_structure)

    def forward(self, target, image):
        z,jac = self.cinn(target, c=self.cond_net(image))
        return z, jac

    def reverse_sample(self, z, image):
        return self.cinn(z, c=self.cond_net(image), rev=True)

    def load(self, name):
        loaded = torch.load(name)
        self.cinn.load_state_dict(loaded['cinn_state_dict'], strict=False)
        self.cond_net.load_state_dict(loaded['cond_net_state_dict'])
        self.optimizer.load_state_dict(loaded['optim_state_dict'])
        epoch = loaded['epoch']
        time = loaded['time']
        return epoch, time


    def save(self, epoch, time, name):
        torch.save({
                    'cinn_state_dict':self.cinn.state_dict(),
                    'cond_net_state_dict':self.cond_net.state_dict(),
                    'optim_state_dict':self.optimizer.state_dict(),
                    'epoch':epoch,
                    'time':time
                    }, name)
