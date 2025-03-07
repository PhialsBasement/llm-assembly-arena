import time
import json
import logging
from colorama import Fore, Style

logger = logging.getLogger('LLM-ASM-Battle')

class AsmBattleGame:
    """Main game logic for the assembly battle game"""

    def __init__(self, claude_player, gpt_player, max_rounds=30):
        self.claude = claude_player
        self.gpt = gpt_player

        self.current_round = 0
        self.setup_rounds = 2  # Allow 2 rounds for defensive setup
        self.max_rounds = max_rounds
        self.winner = None
        self.battle_log = []
        
        # Memory impact tracking
        self.claude_memory_impact = {}
        self.gpt_memory_impact = {}
        
        # Attack budgeting system
        self.max_attack_budget = 1000
        self.claude_attack_budget = self.max_attack_budget
        self.gpt_attack_budget = self.max_attack_budget
        self.budget_regen_rate = 200  # Budget points regenerated per round
        
        # Attack cost definitions
        self.attack_costs = {
            'basic': 50,      # Simple attacks that modify few memory locations
            'targeted': 150,  # Attacks targeting specific memory regions
            'complex': 300,   # Complex attacks with multiple phases or strategies
            'advanced': 450   # Advanced attacks with sophisticated techniques
        }

    def _load_code_to_memory(self, code, cpu):
        """Convert assembly code to bytes and load into CPU memory"""
        # Simple encoding: just store each character as a byte
        code_bytes = [ord(c) for c in code]
        cpu.load_program(code_bytes)

    def _format_battle_log(self, round_num, attacker, defender, attack_code, execution_result, attack_cost, remaining_budget):
        """Format battle log entry"""
        corrupted = len(defender.cpu.corrupted_addresses)
        error = execution_result.get('error', 'None')

        log = f"""
{Fore.YELLOW}============== ROUND {round_num} =============={Style.RESET_ALL}
{Fore.CYAN}{attacker.name}{Style.RESET_ALL} attacks {Fore.MAGENTA}{defender.name}{Style.RESET_ALL}!

{Fore.GREEN}Attack Code:{Style.RESET_ALL}
{attack_code}

{Fore.RED}Execution Result:{Style.RESET_ALL}
- Instructions executed: {execution_result.get('instructions_executed', 0)}
- Memory locations corrupted: {corrupted}
- Error: {error}
- Attack cost: {attack_cost} points
- Remaining budget: {remaining_budget} points

{Fore.BLUE}Memory Dump After Attack:{Style.RESET_ALL}
{defender.cpu.dump_memory(0, 64)}

{Fore.YELLOW}---------------{Style.RESET_ALL}
"""
        return log

    def _update_memory_impact(self, defender):
        """Update memory impact mapping for the defender"""
        if defender.name == "Claude":
            impact_map = self.claude_memory_impact
        else:
            impact_map = self.gpt_memory_impact
            
        # Map memory regions to functionality
        for addr in defender.cpu.corrupted_addresses:
            # Memory region 0-256: Basic operations
            if 0 <= addr < 256:
                impact_map['basic_ops'] = impact_map.get('basic_ops', 0) + 1
            # Memory region 256-512: Loop operations
            elif 256 <= addr < 512:
                impact_map['loops'] = impact_map.get('loops', 0) + 1
            # Memory region 512-768: Memory operations
            elif 512 <= addr < 768:
                impact_map['memory_ops'] = impact_map.get('memory_ops', 0) + 1
            # Memory region 768-1024: Advanced operations
            elif 768 <= addr < 1024:
                impact_map['advanced_ops'] = impact_map.get('advanced_ops', 0) + 1

    def _get_defender_limitations(self, defender):
        """Get limitations based on memory regions corrupted"""
        if defender.name == "Claude":
            impact_map = self.claude_memory_impact
        else:
            impact_map = self.gpt_memory_impact
            
        limitations = []
        
        # Apply limitations based on corruption levels
        basic_ops_impact = impact_map.get('basic_ops', 0)
        if basic_ops_impact > 50:
            limitations.append("All arithmetic operations limited to half effectiveness")
        elif basic_ops_impact > 20:
            limitations.append("Addition/subtraction operations limited")
            
        loops_impact = impact_map.get('loops', 0)
        if loops_impact > 40:
            limitations.append("Loop instructions completely disabled")
        elif loops_impact > 15:
            limitations.append("Loops limited to 10 iterations maximum")
            
        memory_impact = impact_map.get('memory_ops', 0)
        if memory_impact > 30:
            limitations.append("Only first 512 bytes of memory accessible")
        elif memory_impact > 10:
            limitations.append("Memory write operations reduced effectiveness")
            
        advanced_impact = impact_map.get('advanced_ops', 0)
        if advanced_impact > 20:
            limitations.append("Advanced instructions (CALL, INT, etc.) disabled")
            
        return limitations
    
    def _calculate_attack_cost(self, attack_code, corrupted_addresses):
        """Calculate the cost of an attack based on its complexity and impact"""
        # Base cost starts with 'basic'
        base_cost = self.attack_costs['basic']
        
        # Count number of instructions
        instruction_count = len([line for line in attack_code.split('\n') if line.strip() and not line.strip().endswith(':')])
        
        # Check for targeting of specific memory regions
        targets_256_512 = "256" in attack_code and "512" in attack_code
        targets_512_768 = "512" in attack_code and "768" in attack_code
        targets_768_1024 = "768" in attack_code and "1024" in attack_code
        
        # Check for complex control structures
        has_multiple_loops = attack_code.count("loop") > 1 or attack_code.count("LOOP") > 1
        has_conditional_jumps = "JE" in attack_code or "JNE" in attack_code or "JL" in attack_code
        
        # Check for sophisticated instructions
        has_advanced_instructions = "XOR" in attack_code or "SHL" in attack_code or "ROL" in attack_code
        
        # Determine attack complexity
        if len(corrupted_addresses) > 100 or (has_multiple_loops and has_advanced_instructions and targets_768_1024):
            cost = self.attack_costs['advanced']
        elif len(corrupted_addresses) > 50 or (has_conditional_jumps and (targets_256_512 or targets_512_768)):
            cost = self.attack_costs['complex']
        elif len(corrupted_addresses) > 20 or targets_256_512 or targets_512_768 or targets_768_1024:
            cost = self.attack_costs['targeted']
        else:
            cost = self.attack_costs['basic']
            
        # Add instruction count bonus (more instructions = more expensive)
        cost += min(150, instruction_count * 5)
        
        return cost

    def play_setup_round(self):
        """Play setup round for defensive preparations"""
        self.current_round += 1
        
        print(f"\n{Fore.YELLOW}============== SETUP ROUND {self.current_round} =============={Style.RESET_ALL}")
        
        # Claude's setup
        print(f"\n{Fore.CYAN}Claude is setting up defenses...{Style.RESET_ALL}")
        defense_code = self.claude.generate_defense(self.current_round)
        print(f"\n{Fore.GREEN}Claude's Defense Code:{Style.RESET_ALL}")
        print(defense_code)
        
        # Store defense code
        self.claude.defenses.append(defense_code)
        
        # GPT's setup
        print(f"\n{Fore.MAGENTA}GPT is setting up defenses...{Style.RESET_ALL}")
        defense_code = self.gpt.generate_defense(self.current_round)
        print(f"\n{Fore.GREEN}GPT's Defense Code:{Style.RESET_ALL}")
        print(defense_code)
        
        # Store defense code
        self.gpt.defenses.append(defense_code)
        
        # Load defenses into memory
        self._load_code_to_memory(defense_code, self.claude.cpu)
        result = self.claude.cpu.execute()
        print(f"\n{Fore.CYAN}Claude's defense setup complete. Instructions executed: {result.get('instructions_executed', 0)}{Style.RESET_ALL}")
        
        self._load_code_to_memory(defense_code, self.gpt.cpu)
        result = self.gpt.cpu.execute()
        print(f"\n{Fore.MAGENTA}GPT's defense setup complete. Instructions executed: {result.get('instructions_executed', 0)}{Style.RESET_ALL}")
        
        # Add a short delay
        time.sleep(1)

    def play_round(self):
        """Play one round of the battle"""
        self.current_round += 1

        # Determine who goes first (alternating)
        if self.current_round % 2 == 1:
            attacker, defender = self.claude, self.gpt
            attacker_budget = self.claude_attack_budget
        else:
            attacker, defender = self.gpt, self.claude
            attacker_budget = self.gpt_attack_budget

        print(f"\n{Fore.YELLOW}============== ROUND {self.current_round} =============={Style.RESET_ALL}")
        print(f"{Fore.CYAN}{attacker.name}{Style.RESET_ALL} is attacking {Fore.MAGENTA}{defender.name}{Style.RESET_ALL}!")
        print(f"Attack Budget: {attacker_budget} points")

        # Get defender limitations
        limitations = self._get_defender_limitations(defender)
        if limitations:
            print(f"\n{Fore.RED}Defender Limitations:{Style.RESET_ALL}")
            for limitation in limitations:
                print(f"- {limitation}")

        # Generate attack with budget constraints
        print(f"\n{Fore.CYAN}Generating attack code...{Style.RESET_ALL}")
        attack_code = attacker.generate_attack(defender, self.current_round, limitations, attacker_budget)

        print(f"\n{Fore.GREEN}Attack Code:{Style.RESET_ALL}")
        print(attack_code)

        # Load attack code into defender's CPU
        self._load_code_to_memory(attack_code, defender.cpu)

        # Execute attack
        print(f"\n{Fore.RED}Executing attack...{Style.RESET_ALL}")
        execution_result = defender.cpu.execute()

        # Calculate damage (number of corrupted memory locations)
        corrupted = len(defender.cpu.corrupted_addresses)
        
        # Calculate attack cost
        attack_cost = self._calculate_attack_cost(attack_code, defender.cpu.corrupted_addresses)
        
        # Apply budget constraints
        if attack_cost > attacker_budget:
            # If attack costs more than available budget, reduce effectiveness
            reduction_factor = attacker_budget / attack_cost
            original_corrupted = corrupted
            corrupted = int(corrupted * reduction_factor)
            print(f"\n{Fore.RED}Attack exceeds budget! Effectiveness reduced to {int(reduction_factor*100)}%{Style.RESET_ALL}")
            print(f"Original corruption: {original_corrupted}, Reduced to: {corrupted}")
            attack_cost = attacker_budget  # Use all available budget
            
        # Update attacker's budget
        if attacker.name == "Claude":
            self.claude_attack_budget -= attack_cost
        else:
            self.gpt_attack_budget -= attack_cost

        # Update memory impact tracking
        self._update_memory_impact(defender)

        print(f"\n{Fore.MAGENTA}Attack Results:{Style.RESET_ALL}")
        print(f"- Instructions executed: {execution_result.get('instructions_executed', 0)}")
        print(f"- Memory locations corrupted: {corrupted}")
        print(f"- Error: {execution_result.get('error', 'None')}")
        print(f"- Attack cost: {attack_cost} points")
        print(f"- Remaining budget: {attacker.name} - {self.claude_attack_budget if attacker.name == 'Claude' else self.gpt_attack_budget}")

        # Show memory dump after attack
        print(f"\n{Fore.BLUE}Memory Dump After Attack:{Style.RESET_ALL}")
        print(defender.cpu.dump_memory(0, 64))

        # Update score
        attacker.score += corrupted
        
        # Regenerate some budget for both players every round
        self.claude_attack_budget = min(self.max_attack_budget, self.claude_attack_budget + self.budget_regen_rate)
        self.gpt_attack_budget = min(self.max_attack_budget, self.gpt_attack_budget + self.budget_regen_rate)

        # Add to battle log
        log_entry = self._format_battle_log(
            self.current_round, attacker, defender, attack_code, execution_result, 
            attack_cost, self.claude_attack_budget if attacker.name == 'Claude' else self.gpt_attack_budget
        )
        self.battle_log.append(log_entry)

        # Reset corrupted addresses for next round
        defender.cpu.corrupted_addresses = set()

        # Do NOT reinitialize memory so that damage persists
        # defender.cpu._initialize_memory() - commented out

        return {
            'round': self.current_round,
            'attacker': attacker.name,
            'defender': defender.name,
            'corrupted': corrupted,
            'attack_cost': attack_cost,
            'attacker_score': attacker.score,
            'defender_score': defender.score,
            'claude_budget': self.claude_attack_budget,
            'gpt_budget': self.gpt_attack_budget
        }

    def play_game(self):
        """Play the complete game"""
        print(f"{Fore.YELLOW}========================================{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}== LLM ASSEMBLY BATTLE ARENA STARTED =={Style.RESET_ALL}")
        print(f"{Fore.YELLOW}========================================{Style.RESET_ALL}")
        print(f"\n{Fore.CYAN}Claude{Style.RESET_ALL} vs {Fore.MAGENTA}GPT{Style.RESET_ALL}")
        print(f"Setup Rounds: {self.setup_rounds}")
        print(f"Battle Rounds: {self.max_rounds}")
        print(f"Starting Attack Budget: {self.max_attack_budget} points")
        print(f"Budget Regeneration: {self.budget_regen_rate} points per round")
        
        # Play setup rounds
        print(f"\n{Fore.YELLOW}====== DEFENSIVE SETUP PHASE ======{Style.RESET_ALL}")
        for _ in range(self.setup_rounds):
            self.play_setup_round()
        
        # Reset round counter for battle phase
        self.current_round = 0
        
        print(f"\n{Fore.YELLOW}====== BATTLE PHASE BEGINS ======{Style.RESET_ALL}")

        # Play all rounds
        for _ in range(self.max_rounds):
            self.play_round()

            # Show current score
            print(f"\n{Fore.YELLOW}Current Score:{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Claude:{Style.RESET_ALL} {self.claude.score}")
            print(f"{Fore.MAGENTA}GPT:{Style.RESET_ALL} {self.gpt.score}")
            
            # Show budget status
            print(f"\n{Fore.YELLOW}Attack Budgets:{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Claude:{Style.RESET_ALL} {self.claude_attack_budget} points")
            print(f"{Fore.MAGENTA}GPT:{Style.RESET_ALL} {self.gpt_attack_budget} points")
            
            # Show memory impact
            print(f"\n{Fore.YELLOW}Memory Impact:{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Claude Memory Regions Affected:{Style.RESET_ALL} {self.claude_memory_impact}")
            print(f"{Fore.MAGENTA}GPT Memory Regions Affected:{Style.RESET_ALL} {self.gpt_memory_impact}")

            # Add a short delay between rounds
            time.sleep(1)

        # Determine winner
        if self.claude.score > self.gpt.score:
            self.winner = self.claude
        elif self.gpt.score > self.claude.score:
            self.winner = self.gpt
        else:
            self.winner = None

        # Show final results
        self._show_results()

        # Save battle log to file
        with open("battle_log.txt", "w") as f:
            f.write("\n".join(self.battle_log))

        # Save attack and defense code history
        self._save_history()

    def _show_results(self):
        """Show the final results of the battle"""
        print(f"\n{Fore.YELLOW}========================================{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}======== BATTLE RESULTS ========{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}========================================{Style.RESET_ALL}")

        print(f"\n{Fore.CYAN}Claude Score:{Style.RESET_ALL} {self.claude.score}")
        print(f"{Fore.MAGENTA}GPT Score:{Style.RESET_ALL} {self.gpt.score}")
        
        print(f"\n{Fore.CYAN}Claude Final Budget:{Style.RESET_ALL} {self.claude_attack_budget}")
        print(f"{Fore.MAGENTA}GPT Final Budget:{Style.RESET_ALL} {self.gpt_attack_budget}")

        if self.winner:
            winner_color = Fore.CYAN if self.winner.name == "Claude" else Fore.MAGENTA
            print(f"\n{Fore.YELLOW}The winner is: {winner_color}{self.winner.name}{Style.RESET_ALL}!")
        else:
            print(f"\n{Fore.YELLOW}The battle ended in a draw!{Style.RESET_ALL}")

        print(f"\nBattle log saved to 'battle_log.txt'")

    def _save_history(self):
        """Save the history of attacks and defenses to a JSON file"""
        history = {
            "claude_defenses": self.claude.defenses,
            "gpt_defenses": self.gpt.defenses,
            "claude_attacks": self.claude.attacks,
            "gpt_attacks": self.gpt.attacks,
            "claude_score": self.claude.score,
            "gpt_score": self.gpt.score,
            "claude_memory_impact": self.claude_memory_impact,
            "gpt_memory_impact": self.gpt_memory_impact,
            "claude_final_budget": self.claude_attack_budget,
            "gpt_final_budget": self.gpt_attack_budget,
            "winner": self.winner.name if self.winner else "Draw"
        }

        with open("battle_history.json", "w") as f:
            json.dump(history, f, indent=2)

        print(f"Battle history saved to 'battle_history.json'")