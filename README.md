# Project description
- A Reinforcement learning project. Apply DoubleDuelingDeepQ to make a Google's Dinosaur player.

# Demo

  <table>
    <tr align="center">
      <td>
<img src="./demo/DinoRandom.gif"/>
      </td>
      <td>
<img src="./demo/Dino50.gif"/>
      </td>
    </tr>
    <tr align="center">
      <td> 
<img src="./demo/Dino100.gif"/>
      </td>
      <td>
<img src="./demo/Dino250.gif"/>
      </td>
    </tr>
  </table>

<div align='center'><b>Figure 1</b>: Game plays demo. </div>
<div align='center'>(<i>Top-Left</i>) Random player. (<i>Top-Right</i>) Players at 50 episodes/901 game-steps/408 learned-steps. (<i>Bottom-Left</i>) Players at 100 episodes/2360 game-steps/808 learned-steps. (<i>Bottom-Right</i>) Players at 250 episodes/16604 game-steps/3356 learned-steps. Training with <i>learning_rate=0.005, starting_epsilon=1.0, epsilon_dec=10^-3, batch_size=32, memory_size=256, target_network_replace=4</i>.</div>

# Running:
- Open chorme dino game at
    
    
        chrome://dino
- Update screen configuration in *environment\env_config*
- Run 

    
        python host.py  
