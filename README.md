# 🤖 AI Assembly Battle Arena

*Watch AI language models duke it out in x86 assembly code!*

Hey there! Welcome to one of my coolest side projects - a simulated battle arena where large language models (Claude vs GPT) compete by writing assembly code to mess with each other's virtual memory! This project came from a late-night thought: "What if we let AI systems attack each other in low-level code?"

## 💡 What the heck is this?

This is a battle simulation where Claude and GPT take turns writing x86 assembly code to corrupt each other's virtual memory. Each AI controls a simulated CPU with memory regions representing different functionality, and the goal is to corrupt as much of your opponent's memory as possible while protecting your own.

Think of it like a weird competitive coding game with AI - they're literally trying to hack each other! 😄

## 🔥 Key Features

- Complete virtual CPU implementation with support for common x86 assembly instructions
- Budgeting system that limits attack complexity (no infinite attacks!)
- Memory region system where different areas control different functionality
- Defensive setup phase where AIs can protect their memory
- Visualizations of the battle using Manim (the math animation library)
- Battle history tracking and scoring system

## 🛠️ Installation

First, clone this repo:

```bash
git clone https://github.com/phialsbasement/ai-assembly-battle.git
cd ai-assembly-battle
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

You'll need API keys for both Anthropic (Claude) and OpenAI (GPT). Create a `.env` file in the project root with:

```
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here
```

## 🚀 Running a Battle

Just run the main script:

```bash
python main.py
```

The battle will run for 30 rounds by default (configurable in `main.py`), alternating attacks between Claude and GPT. You'll see:
- Defensive setup rounds
- Attack code for each round
- Memory corruption results
- Budgeting information
- Final scores and battle statistics

The output is saved in `battle_log.txt` and `battle_history.json` for later analysis or visualization.

## 🎬 Visualizing Battles

After running a battle, you can create an awesome visualization using:

```bash
python -m manim -pql ai_battle_visualization.py AIBattleSimulation
```

This creates a full animation showing:
- Memory regions and their functions
- Attack sequences and impacts
- Memory corruption effects
- Final battle statistics with colorful charts

*Note: Manim can be a bit finnicky to install - check out [Manim's installation guide](https://docs.manim.community/en/stable/installation.html) if you have issues!*

## 🧠 How the Battle Works

1. **Setup Phase**: Each AI writes defensive code to protect their memory.
2. **Battle Phase**: AIs take turns attacking each other:
   - They analyze opponent's memory
   - Generate assembly code to corrupt it
   - Pay "budget points" based on attack complexity
   - Score points for each corrupted memory address

3. **Memory Regions**: The virtual CPU has four key memory regions:
   - 0-255: Basic operations (arithmetic, etc.)
   - 256-511: Loops and branching
   - 512-767: Memory access
   - 768-1023: Advanced functions

4. **Scoring**: The winner is the AI with the most corrupted opponent memory locations.

## 🔍 Project Structure

- `main.py` - Entry point that sets up and runs the battle
- `src/`
  - `virtual_cpu.py` - Simulated CPU with x86 instruction support
  - `llm_player.py` - Claude and GPT player implementations
  - `asm_battle_game.py` - Main battle game logic
- `ai_battle_visualization.py` - Battle visualization using Manim

## 🤝 Contributing

I'd absolutely love contributions! Whether it's adding new instructions to the CPU, improving the battle mechanics, or creating better visualizations.

Some ideas:
- Add more CPU instructions
- Implement network effects (botnet battles?)
- Create a web UI to watch battles in real-time
- Add more AIs (Llama, Gemini, etc.)

Just open a PR with your awesome changes!

## ⚠️ Disclaimer

This is a completely isolated, simulated environment. No real systems are affected by this code. It's purely for fun, education and to explore AI capabilities in a constrained domain.

## 📝 License

MIT License - go wild with it!
