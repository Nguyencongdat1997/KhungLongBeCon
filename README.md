# Project description
- A Reinforcement learning project. Apply DoubleDuelingDeepQ to make a Google's Dinosaur player.

# Demo

  <table>
    <tr align="center">
      <td>
<figure class="video_container">
  <iframe src="https://user-images.githubusercontent.com/8006495/121773646-6a665380-cba7-11eb-9ecf-2b34e6e7ad0e.mp4" frameborder="0" allowfullscreen="false"> </iframe>
</figure>
      </td>
      <td>
https://user-images.githubusercontent.com/8006495/121773649-7b16c980-cba7-11eb-92f1-e6fded7b6cf9.mp4
      </td>
    </tr>
    <tr align="center">
      <td> 
https://user-images.githubusercontent.com/8006495/121773652-8538c800-cba7-11eb-8b9d-2dab31ee3c37.mp4
      </td>
      <td>
https://user-images.githubusercontent.com/8006495/121773662-91bd2080-cba7-11eb-9138-2eb07dfa4cb9.mp4
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
