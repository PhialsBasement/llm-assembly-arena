#!/usr/bin/env python3
import os
import logging
import anthropic
import openai
from dotenv import load_dotenv
from colorama import init

# Import our game modules
from src.virtual_cpu import VirtualCPU
from src.llm_player import LLMPlayer
from src.asm_battle_game import AsmBattleGame

# Initialize colorama for cross-platform colored terminal output
init(autoreset=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='llm_battle.log'
)
logger = logging.getLogger('LLM-ASM-Battle')

def main():
    """Main function to run the game"""
    # Load environment variables
    load_dotenv()

    # Get API keys
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    # Check if API keys are set
    if not ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY environment variable not set.")
        print("Please set your Anthropic API key in the .env file or as an environment variable.")
        return

    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key in the .env file or as an environment variable.")
        return

    # Initialize API clients
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

    # Create CPUs for each player
    claude_cpu = VirtualCPU(memory_size=1024, owner="claude")
    gpt_cpu = VirtualCPU(memory_size=1024, owner="gpt")

    # Create players
    claude = LLMPlayer("Claude", anthropic_client, claude_cpu)
    gpt = LLMPlayer("GPT", openai_client, gpt_cpu)

    # Create and play the game with budgeting system
    game = AsmBattleGame(claude, gpt, max_rounds=30)
    try:
        game.play_game()
    except KeyboardInterrupt:
        logger.warning("Game interrupted by user.")
        print("\nGame interrupted by user.")
    except Exception as e:
        logger.error(f"Game error: {str(e)}", exc_info=True)
        print(f"\nError during game: {str(e)}")

if __name__ == "__main__":
    main()