# AINavigator
Projects as part of Udacity Nanodegree - AI Deep Reinforcement Learning 


The Environment
For this project, we will train an agent to navigate (and collect bananas!) in a large, square world in Unity environment.

 <img src="https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif" width="384">


A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

0 - move forward.
1 - move backward.
2 - turn left.
3 - turn right.

The task is episodic, and in order to solve the environment, the trained agent must get an average score of +13 over 100 consecutive episodes.

Source Code & Report 

1. The Navigation.ipynb
    file with fully functional code displaying output to give you an overall program flow start with Unity Environment till Trained agent average score plot
2. Report.pdf :  
    project report in details covering DeepQ and Agent and Model details
3. NetworkWeight.pt 
    A file with the saved model weights of the successful agent, can be named something like model.pt.


Note:
For the detail instruction please check https://github.com/udacity/deep-reinforcement-learning/blob/master/p1_navigation/README.md
To setup, you need Python 3.6, Unity Environment to be installed.Once you setup your python environment , have agent.py ,nn_model.py and checkpoint17.pt if you wish to directly test the trained Agent.

