import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os.path

class Network(nn.Module):
    def __init__(self, Input, Output, FirstNN, MaxNN, SecondNN, LearnRate):
        super(Network, self).__init__()
        self.fc1    = nn.Linear(Input, FirstNN)
        self.fc2    = nn.Linear(FirstNN, MaxNN)
        self.fc3    = nn.Linear(MaxNN, SecondNN)
        self.fc4    = nn.Linear(SecondNN, Output)
        self.relu   = nn.LeakyReLU()
        self.optim  = optim.Adam(self.parameters(), lr = LearnRate)
        self.loss   = nn.SmoothL1Loss()
        self.sched  = optim.lr_scheduler.StepLR(self.optim, gamma = 0.9, step_size = 100)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class EnvAgent:
    def __init__(self, ReplaySize, BatchSize, Input, Output, EpsilonStart, Decay, EpsilonEnd, PolicyNet, TargetNet, Gamma):
        #declare Fundamentals
        self.BatchSize  = BatchSize
        self.Input      = Input
        self.Output     = Output
        self.Pos        = 0
        self.MemSize    = ReplaySize
        self.ActionSpace= [i for i in range(Output)]

        #declare stuff for random decay
        self.EpsStart   = EpsilonStart
        self.EpsDecay   = Decay
        self.EpsEnd     = EpsilonEnd
        self.Eps        = EpsilonStart
        self.Gamma      = Gamma

        #declare buffers
        self.StateMem   = np.zeros((self.MemSize, self.Input), dtype = np.float32)
        self.ActionMem  = np.zeros(self.MemSize, dtype = np.int32)
        self.NextMem    = np.zeros((self.MemSize, self.Input), dtype = np.float32)
        self.TermMem    = np.zeros(self.MemSize, dtype = np.bool)
        self.RewardMem  = np.zeros(self.MemSize, dtype = np.float32)

        #Declare Nets
        self.PolicyNet  = PolicyNet
        self.TargetNet  = TargetNet
        self.TargetNet.eval()
    
    def SaveNet(self):
        torch.save(self.PolicyNet.state_dict(), 'PolicyNet.nn')
        torch.save(self.TargetNet.state_dict(), 'TargetNet.nn')
    
    def LoadNet(self):
        if os.path.isfile('PolicyNet.nn'):
            self.PolicyNet.load_state_dict(torch.load('PolicyNet.nn'))
        if os.path.isfile('TargetNet.nn'):
            self.TargetNet.load_state_dict(torch.load('TargetNet.nn'))
    
    def UpdateTarget(self):
        self.TargetNet.load_state_dict(self.PolicyNet.state_dict())
    
    def SaveCondition(self, State, Action, Reward, NextState, Terminal):
        Index = self.Pos % self.MemSize
        self.StateMem[Index]    = State
        self.ActionMem[Index]   = Action
        self.RewardMem[Index]   = Reward
        self.NextMem[Index]     = NextState
        self.TermMem[Index]     = Terminal
        self.Pos                = self.Pos + 1
    
    def TestTarget(self, State):
        with torch.no_grad():
            Input = torch.tensor(State, dtype = torch.float32)
            return self.TargetNet(Input).argmax(dim = 1)
    
    def TestPolicy(self, State):
        with torch.no_grad():
            Input = torch.tensor([State], dtype = torch.float32)
            return self.PolicyNet(Input).argmax(dim = 1)
    
    def TakeAction(self, State):
        if np.random.random() > self.Eps:
            print('Tree', end = ' ')
            Input = torch.tensor([State], dtype = torch.float32)
            return self.PolicyNet(Input).argmax().item()
        else:
            print('Random', end = ' ')
            return np.random.choice(self.ActionSpace)
    
    def DecayEpsilon(self):
        self.Eps = self.Eps - self.EpsDecay
        if(self.Eps < self.EpsEnd):
            self.Eps = self.EpsEnd
    
    def Train(self):
        if self.Pos < self.BatchSize:
            return
        else:
            #Create Batches
            MaxVal      = min(self.Pos, self.MemSize)
            Batch       = np.random.choice(MaxVal, self.BatchSize, replace = False)
            BatchIndex  = np.arange(self.BatchSize, dtype = np.int32)
            State       = torch.tensor(self.StateMem[Batch])
            NextState   = torch.tensor(self.NextMem[Batch])
            Reward      = torch.tensor(self.RewardMem[Batch])
            Terminal    = torch.tensor(self.TermMem[Batch])
            Action      = self.ActionMem[Batch]

            #Get Gradients and Train
            self.PolicyNet.optim.zero_grad()
            Output  = self.PolicyNet(State)[BatchIndex, Action]
            Next    = self.TargetNet(NextState)
            Next [Terminal] = 0.0   
            #Use BellMan Equation
            #MaxReward = Reward + gamma*MaxFutureAction
            #IncreaseProbability = Reward + gamma * MaxProbAction
            Target = Reward + self.Gamma*torch.max(Next, dim = 1)[0]

            loss = self.PolicyNet.loss(Output, Target)
            loss.backward()
            self.PolicyNet.optim.step() 
            self.PolicyNet.sched.step()
            self.DecayEpsilon()
            return loss