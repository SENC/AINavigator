#!/usr/bin/env python
# coding: utf-8

# # Navigation
# 
# ---
# 
# You are welcome to use this coding environment to train your agent for the project.  Follow the instructions below to get started!
# 
# ### 1. Start the Environment
# 
# Run the next code cell to install a few packages.  This line will take a few minutes to run!

# In[2]:


#Reference Libraries
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import random
import torch


# In[3]:


get_ipython().system('pip -q install ./python')


# The environment is already saved in the Workspace and can be accessed at the file path provided below.  Please run the next code cell without making any changes.

# In[4]:


from unityagents import UnityEnvironment
import numpy as np

# please do not modify the line below
env = UnityEnvironment(file_name="/data/Banana_Linux_NoVis/Banana.x86_64")


# Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.

# In[5]:


# get the default brain
brain_name = env.brain_names[0]
print(brain_name)
brain = env.brains[brain_name]
brain
print(brain)


# 

# In[6]:


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)


# ### 3. Take Random Actions in the Environment
# 
# In the next code cell, you will learn how to use the Python API to control the agent and receive feedback from the environment.
# 
# Note that **in this coding environment, you will not be able to watch the agent while it is training**, and you should set `train_mode=True` to restart the environment.

# In[6]:


env_info = env.reset(train_mode=True)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state
score = 0                                          # initialize the score
while True:
    action = np.random.randint(action_size)        # select an action
    #action = agent.act(state)
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    if done:                                       # exit loop if episode finished
        break
    
print("Score: {}".format(score))


# ### 4. Instantiate AI Agent and Train  in the Environment 
# 
# 

# In[7]:


#DeepQLearning 
from nn_model import QDeepNetwork
from agent import AiAgent


# In[8]:


#Test to create the instance of AiAgent
firstAI = AiAgent(state_size=37,action_size=4,seed=17)


# In[9]:


# Make the Agent interacted with its Environment in single Episode -1000 steps
def test_run_single_episode (env: UnityEnvironment,brain_name, ai_agent:AiAgent=None,max_t=1000,epsl=0.,train_mode=False):
    """Input env|agent|number of steps|test mode
       Output Agent score
    
    """
    env_info = env.reset(train_mode=train_mode)[brain_name]
    action_size = env.brains[brain_name].vector_action_space_size
    state = env_info.vector_observations[0]
    
    #initial the score 
    score =0
    
    #Run each step in the episode
    for _ in range (max_t):
        action = ai_agent.act(state,epsl) if ai_agent else np.random.randint(action_size)
        
        #lets make the agent take the action 
        env_info = env.step(action)[brain_name]
        
        #get the next state 'St+1'
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        
        if ai_agent and train_mode: #Train mode is True then make the agent learn
            ai_agent.step(state, action, reward, next_state, done)
        
        state = next_state
        score += reward
        if done:
            break
    return score

#Train the Agent - DeepQ Network Algorithm
#Function dqn helps to calculate Agent's optimized score to choose Greedy Policy based on rewards
def dqn(ai_agent,num_episodes=1000, max_t=1000, eps_start=0.78, eps_end=0.0025, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    
    
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    

    for i_episode in range(1,num_episodes+1):
        #reset the unity Env_info for each new episode
        env_info = env.reset(train_mode=True)[brain_name]
    
        #set initial state 
        state = env_info.vector_observations[0]
    
        score = 0 
        #Iterative run until all number of episodes done
    
        while True:
            action = ai_agent.act(state,eps)
        
            env_info = env.step(action)[brain_name]        #Take action based on Epsilion Greedy Policy
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished 
            
            ai_agent.step(state, action, reward, next_state, done) # Replay Memory - Learn from potential failure
            
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step            
            if done:                                       # exit loop if episode finished
                break
            
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=17.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(ai_agent.qnetwork_local.state_dict(), 'checkpointai17.pth')
            break
        
    return scores      
       


# In[11]:


#v9.4 -Highy score so repeat once  to get threshold 15
firstAI1 = AiAgent(state_size=37,action_size=4,seed=24)

num_episodes=10000
max_t=1000
eps_start=0.9786 
eps_end=0.0005 
eps_decay=0.9799
scores = dqn(firstAI1,num_episodes,max_t,eps_start,eps_end,eps_decay)


# In[12]:


# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


# In[10]:


#Watch the train Agent perfromance
# load the weights from file
firstAI_trained = AiAgent(state_size=37,action_size=4,seed=24)
firstAI_trained.qnetwork_local.load_state_dict(torch.load('checkpointai17.pth'))
          


# In[11]:


# Test Run score

score = test_run_single_episode(env, brain_name, firstAI_trained)

print(f'Score: {score}')


# ## Check the Trained Agent Performance 
# 
# Based on the training weight, run the agent in the same environment in Test Mode (train_mode= False) and Goal to achieve 13+ as an average score

# In[14]:


scores_window = deque(maxlen=50)
for test_episode in range(1,51):
    score = test_run_single_episode(env, brain_name, firstAI_trained)
    print(f'Score: {score}')    
    scores_window.append(score)       # save most recent score
    print('\rEpisode {}\tAverage Score: {:.2f}\t'.format(test_episode, np.mean(scores_window)), end= "")


# In[13]:


for _ in range (10):
    score = test_run_single_episode(env, brain_name, firstAI_trained)
    print(f'Score: {score}')


# In[17]:


env.close()


# #History - Parameter tuning 

# In[45]:


#v9.3 -Highy score so repeat once  to get threshold 15
firstAI1 = AiAgent(state_size=37,action_size=4,seed=24)

num_episodes=500
max_t=1000
eps_start=0.976 
eps_end=0.0005 
eps_decay=0.979
scores = dqn(firstAI1,num_episodes,max_t,eps_start,eps_end,eps_decay)


# In[42]:


#v9 -Highy score so repeat once  to get threshold 17
firstAI1 = AiAgent(state_size=37,action_size=4,seed=24)

num_episodes=1000
max_t=1000
eps_start=0.976 
eps_end=0.0005 
eps_decay=0.979
scores = dqn(firstAI1,num_episodes,max_t,eps_start,eps_end,eps_decay)


# In[33]:


#v9 -Highy score so repeat once  
firstAI1 = AiAgent(state_size=37,action_size=4,seed=24)

num_episodes=1000
max_t=1000
eps_start=0.976 
eps_end=0.0005 
eps_decay=0.979
scores = dqn(firstAI1,num_episodes,max_t,eps_start,eps_end,eps_decay)


# In[31]:


#v11  Updated EveryTimeStep to 4 
firstAI1 = AiAgent(state_size=37,action_size=4,seed=24)

num_episodes=1000
max_t=500
eps_start=0.786 
eps_end=0.0005 
eps_decay=0.9979
scores = dqn(firstAI1,num_episodes,max_t,eps_start,eps_end,eps_decay)


# In[30]:


#v10  Updated EveryTimeStep to 4 
firstAI1 = AiAgent(state_size=37,action_size=4,seed=24)

num_episodes=1000
max_t=1000
eps_start=0.976 
eps_end=0.0005 
eps_decay=0.979
scores = dqn(firstAI1,num_episodes,max_t,eps_start,eps_end,eps_decay)


# In[28]:


#v9  Updated EveryTimeStep to 4 
firstAI1 = AiAgent(state_size=37,action_size=4,seed=24)

num_episodes=500
max_t=1000
eps_start=0.976 
eps_end=0.0005 
eps_decay=0.979
scores = dqn(firstAI1,num_episodes,max_t,eps_start,eps_end,eps_decay)


# In[26]:


#v7  Updated EveryTimeStep to 4 
firstAI1 = AiAgent(state_size=37,action_size=4,seed=24)

num_episodes=500
max_t=1000
eps_start=0.917 
eps_end=0.0007 
eps_decay=0.979
scores = dqn(firstAI1,num_episodes,max_t,eps_start,eps_end,eps_decay)


# In[25]:


#v7  Updated EveryTimeStep to 4 
firstAI1 = AiAgent(state_size=37,action_size=4,seed=24)

num_episodes=300
max_t=1000
eps_start=0.8217 
eps_end=0.0027 
eps_decay=0.979
scores = dqn(firstAI1,num_episodes,max_t,eps_start,eps_end,eps_decay)


# In[23]:


#v6  Updated EveryTimeStep to 4 
firstAI1 = AiAgent(state_size=37,action_size=4,seed=24)

num_episodes=1000
max_t=1000
eps_start=0.7817 
eps_end=0.0017 
eps_decay=0.919
scores = dqn(firstAI1,num_episodes,max_t,eps_start,eps_end,eps_decay)


# In[18]:


#v6  Updated EveryTimeStep to 4 
firstAI1 = AiAgent(state_size=37,action_size=4,seed=24)

num_episodes=1000
max_t=1000
eps_start=0.7817 
eps_end=0.0025 
eps_decay=0.919
scores = dqn(firstAI1,num_episodes,max_t,eps_start,eps_end,eps_decay)


# In[17]:


#v6  Updated EveryTimeStep to 4 
firstAI1 = AiAgent(state_size=37,action_size=4,seed=1)

num_episodes=300
max_t=1000
eps_start=0.6917 
eps_end=0.0015 
eps_decay=0.999
scores = dqn(firstAI1,num_episodes,max_t,eps_start,eps_end,eps_decay)


# In[14]:


#v5  Updated EveryTimeStep to 4 
firstAI1 = AiAgent(state_size=37,action_size=4,seed=1)

num_episodes=300
max_t=1000
eps_start=0.24 
eps_end=0.0025 
eps_decay=0.895
scores = dqn(firstAI1,num_episodes,max_t,eps_start,eps_end,eps_decay)


# In[12]:


#v4  Updated EveryTimeStep to 4 
firstAI1 = AiAgent(state_size=37,action_size=4,seed=1)

num_episodes=300
max_t=1000
eps_start=0.60 
eps_end=0.0025 
eps_decay=0.895
scores = dqn(firstAI1,num_episodes,max_t,eps_start,eps_end,eps_decay)


# In[11]:


#v3  Updated EveryTimeStep to 4 and eps start .78 ep end 0.0025 seed 24
firstAI1 = AiAgent(state_size=37,action_size=4,seed=24)

scores = dqn(firstAI1)


# In[10]:


#v2  Updated EveryTimeStep to 4 and eps start .78 ep end 0.0025 seed 24
firstAI1 = AiAgent(state_size=37,action_size=4,seed=24)

scores = dqn(firstAI1)


# In[ ]:


#v2  Updated EveryTimeStep to 4 and eps start .79 seed 24
firstAI1 = AiAgent(state_size=37,action_size=4,seed=17)

scores = dqn(firstAI1)


# In[12]:


firstAI1 = AiAgent(state_size=37,action_size=4,seed=24)

scores = dqn(firstAI1)


# In[ ]:


#v3  Updated EveryTimeStep to 4 and 
firstAI1 = AiAgent(state_size=37,action_size=4,seed=17)

scores = dqn(firstAI1)


# In[34]:


# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


# In[35]:


#Watch the train Agent perfromance
# load the weights from file
firstAI_trained = AiAgent(state_size=37,action_size=4,seed=24)
firstAI_trained.qnetwork_local.load_state_dict(torch.load('checkpointai13.pth'))
          


# In[36]:


def test_run_single_episode (env: UnityEnvironment,brain_name, ai_agent:AiAgent=None,max_t=1000,epsl=0.,train_mode=False):
    """Input env|agent|number of steps|test mode
       Output Agent score
    
    """
    env_info = env.reset(train_mode=train_mode)[brain_name]
    action_size = env.brains[brain_name].vector_action_space_size
    state = env_info.vector_observations[0]
    
    #initial the score 
    score =0
    
    #Run each step in the episode
    for _ in range (max_t):
        action = ai_agent.act(state,epsl) if ai_agent else np.random.randint(action_size)
        
        #lets make the agent take the action 
        env_info = env.step(action)[brain_name]
        
        #get the next state 'St+1'
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        
        if ai_agent and train_mode: #Train mode is True then make the agent learn
            ai_agent.step(state, action, reward, next_state, done)
        
        state = next_state
        score += reward
        if done:
            break
    return score
        
            


# In[38]:


# Test Run score
score = test_run_single_episode(env, brain_name, firstAI_trained)

print(f'Score: {score}')  


# In[40]:


# Test Run score
score = test_run_single_episode(env, brain_name, firstAI_trained)

print(f'Score: {score}')


# When finished, you can close the environment.

# In[ ]:


env.close()


# ### 4. It's Your Turn!
# 
# Now it's your turn to train your own agent to solve the environment!  A few **important notes**:
# - When training the environment, set `train_mode=True`, so that the line for resetting the environment looks like the following:
# ```python
# env_info = env.reset(train_mode=True)[brain_name]
# ```
# - To structure your work, you're welcome to work directly in this Jupyter notebook, or you might like to start over with a new file!  You can see the list of files in the workspace by clicking on **_Jupyter_** in the top left corner of the notebook.
# - In this coding environment, you will not be able to watch the agent while it is training.  However, **_after training the agent_**, you can download the saved model weights to watch the agent on your own machine! 
