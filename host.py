from environment.game import Game
from players.random_player import RandomPlayer

if __name__ == "__main__":
    player = RandomPlayer()
    game = Game(player)
    game.run()
