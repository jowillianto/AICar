import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
channel = EngineConfigurationChannel()
env = UnityEnvironment(file_name = 'Road1/Prototype 1', side_channels=[channel])
from TorchQ import Network, EnvAgent

env.reset()
behavior_name = list(env.behavior_specs)[0]

channel.set_configuration_parameters(time_scale = 5)

Motor      = [0, 150, 150]
Act     = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def GetSensor(cur_obs):
    return list(cur_obs[6:12])

def GetPosition(cur_obs):
    return list(cur_obs[0:3])

def GetFinish(cur_obs):
    return list(cur_obs[3:6])

def SetMotor(action):
    global Motor
    Motor[0] = Act[action]

def Read():
    decision_steps, _ = env.get_steps(behavior_name)
    cur_obs           = decision_steps.obs[0][0,:]
    return cur_obs

def Distance(pos1, pos2):
    return (pos1[0] - pos2[0])**2 + (pos1[2] - pos2[2])**2

def Dead(pos, accur):
    if(Distance(init_pos, pos) < accur):
        return True
    else:
        return False

#Set Variables
PolicyNet = Network(50, 22, 32, 32, 32, 0.01)
TargetNet = Network(50, 22, 32, 32, 32, 0.01)
Agent     = EnvAgent(10000, 512, 50, 21, 1.0, 0.0005, 0.1, PolicyNet, TargetNet, 0.99)
Term      = False
Reward    = 0
Observe   = Read()
OldPos    = GetPosition(Observe)
State     = GetSensor(Observe)
init_pos  = GetPosition(Observe)
FinishPos = GetFinish(Observe)
NextState = []

for i in range(0, 10):
    NextState.extend(State)
State = NextState.copy()

for i in range(20):
    Action = Agent.TakeAction(State)
    env.set_actions(behavior_name, np.array([Motor]))
    env.step()
    Observe = Read()
    if(i >= 10):
        for j in range(5):
            NextState.pop(0)
        NextState.extend(GetSensor(Observe))
    if(i < 10):
        for j in range(5):
            State.pop(0)
        State.extend(GetSensor(Observe))

for i in range(1000000):
    #Get the Actions
    Action = Agent.TakeAction(State)
    SetMotor(Action)
    
    # Set the actions
    env.set_actions(behavior_name, np.array([Motor]))

    # Move the simulation forward
    env.step()
    Observe     = Read()
    for j in range(5):
        NextState.pop(0)

    NextState.extend(GetSensor(Observe))
    Pos         = GetPosition(Observe)
    #Reward Policy
    MinBool     = min(NextState[40:45]) > 6.8
    CritBool    = min(NextState[40:45]) > 6
    DeadBool    = min(NextState[40:45]) < 6
    MaxBool     = max(NextState[40:45]) > 12
    Dist        = Distance(OldPos, Pos)

    if(MaxBool and MinBool):
        Reward = 30*Dist+3
    elif(CritBool):
        Reward = 20*Dist+2
    elif(DeadBool):
        Reward = 1
    else:
        Reward = 10*Dist+1

    #DeadState
    if(Dead(Pos, 0.05)):
        Reward = -10
        env.reset()
        Motor[0] = 0
        env.set_actions(behavior_name, np.array([Motor]))
        for i in range(0, 3):
            env.step() 
        Term = True
        print('Reset')
        
    if(Distance(Pos, FinishPos) < 150):
        print("Reset")
        Motor[0] = 0
        Term = True
        for i in range(100):
            env.set_actions(behavior_name, np.array([Motor]))
        env.reset()
        break

    for j in range(5):
        State.pop(0)
    State.extend(NextState[40:45])
    Agent.SaveCondition(State, Action, Reward, NextState, Term)
    Loss    = Agent.Train()
    OldPos  = Pos

    if(i % 100 == 0):
        Agent.SaveNet()
    if(i % 10 == 0):
        print('\n',i, ' ', Loss, ' ', Reward,' ')

    #SoftUpdate
    if(i % 1000 == 0):
        Agent.UpdateTarget()
        Agent.SaveNet()
    Term = False
env.close()
