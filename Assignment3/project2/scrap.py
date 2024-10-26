from pacman import GameState
from multiAgents import MinimaxAgent, AlphaBetaAgent

def test_agents():
    # Initialize the game state for the minimaxClassic layout
    gameState = GameState()  # You may need to specify the layout if required

    # Create instances of the agents
    minimax_agent = MinimaxAgent()
    alpha_beta_agent = AlphaBetaAgent()

    # Specify the expected minimax values for depths 1, 2, 3, and 4
    expected_values = {
        1: 9,
        2: 8,
        3: 7,
        4: -492
    }

    # Test each depth
    for depth in range(1, 5):
        # Get actions for both agents
        minimax_action = minimax_agent.getAction(gameState)
        alpha_beta_action = alpha_beta_agent.getAction(gameState)

        # Here you can manually set the depth in your agents if necessary
        # For instance, if your agents have depth parameters, you might need to adjust them accordingly

        # Print the actions taken
        print(f"Depth: {depth}")
        print(f"Minimax Agent Action: {minimax_action}")
        print(f"AlphaBeta Agent Action: {alpha_beta_action}")

        # For this test, we need to simulate the game state after taking actions
        # You need to generate successor states
        minimax_successor = gameState.generateSuccessor(0, minimax_action)  # Pacman's index is 0
        alpha_beta_successor = gameState.generateSuccessor(0, alpha_beta_action)

        # Get the evaluation values for both agents
        minimax_value = minimax_agent.evaluationFunction(minimax_successor)
        alpha_beta_value = alpha_beta_agent.evaluationFunction(alpha_beta_successor)

        # Print the values returned
        print(f"Minimax Value: {minimax_value}, AlphaBeta Value: {alpha_beta_value}")

        # Check if the values match the expected values
        if minimax_value == expected_values[depth]:
            print("Minimax value is correct.")
        else:
            print(f"Minimax value is incorrect. Expected: {expected_values[depth]}, Got: {minimax_value}")

        if alpha_beta_value == expected_values[depth]:
            print("AlphaBeta value is correct.")
        else:
            print(f"AlphaBeta value is incorrect. Expected: {expected_values[depth]}, Got: {alpha_beta_value}")

# Run the test
test_agents()
