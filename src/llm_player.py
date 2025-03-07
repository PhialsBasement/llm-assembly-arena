import logging
import anthropic
import openai
import json
import random

logger = logging.getLogger('LLM-ASM-Battle')

class LLMPlayer:
    """Represents an LLM player in the game"""

    def __init__(self, name, api_client, cpu):
        self.name = name
        self.api_client = api_client
        self.cpu = cpu
        self.score = 0
        self.attacks = []
        self.defenses = []

    def generate_attack(self, opponent, round_num, defender_limitations=None, attack_budget=1000):
        """Generate assembly code to attack the opponent with budget constraints"""
        if self.name.lower() == "claude":
            return self._generate_claude_attack(opponent, round_num, defender_limitations, attack_budget)
        else:
            return self._generate_gpt_attack(opponent, round_num, defender_limitations, attack_budget)
            
    def generate_defense(self, round_num):
        """Generate defensive assembly code"""
        if self.name.lower() == "claude":
            return self._generate_claude_defense(round_num)
        else:
            return self._generate_gpt_defense(round_num)

    def _format_history(self, opponent):
        """Format complete history of moves for both players"""
        history = ""
        
        # Format defenses
        if self.defenses:
            history += "YOUR DEFENSIVE SETUP:\n"
            for i, defense in enumerate(self.defenses):
                history += f"Setup Round {i+1}:\n{defense}\n\n"
        
        if opponent.defenses:
            history += "OPPONENT'S DEFENSIVE SETUP:\n"
            for i, defense in enumerate(opponent.defenses):
                history += f"Setup Round {i+1}:\n{defense}\n\n"
        
        # Format attack history in chronological order
        if self.attacks or opponent.attacks:
            history += "BATTLE HISTORY:\n"
            # Determine maximum rounds
            max_rounds = max(len(self.attacks), len(opponent.attacks))
            
            for i in range(max_rounds):
                # Your attack this round
                if i < len(self.attacks):
                    history += f"Round {i+1} - Your attack:\n{self.attacks[i]}\n\n"
                
                # Opponent's attack this round
                if i < len(opponent.attacks):
                    history += f"Round {i+1} - Opponent's attack:\n{opponent.attacks[i]}\n\n"
        
        return history

    def _generate_budget_info(self, attack_budget):
        """Generate information about different attack types and costs"""
        budget_info = f"""
ATTACK BUDGET INFORMATION:
- Current budget: {attack_budget} points
- Attack costs vary based on complexity and effectiveness:
  - Basic attacks (modify few bytes, simple operations): ~50-100 points
  - Targeted attacks (focused on specific memory regions): ~150-250 points
  - Complex attacks (multiple phases or strategies): ~300-400 points
  - Advanced attacks (sophisticated techniques): ~450-600 points

The cost of your attack will be determined by:
- Number of instructions used
- Number of memory regions targeted
- Complexity of control structures
- Types of instructions used
- Overall impact on opponent's memory

More complex, effective attacks cost more points. If your attack exceeds your budget,
its effectiveness will be reduced proportionally.
"""
        return budget_info

    def _generate_claude_attack(self, opponent, round_num, defender_limitations=None, attack_budget=1000):
        """Generate attack using Claude API with improved error handling and logging"""
        memory_dump = opponent.cpu.dump_memory(0, 128)
        register_dump = opponent.cpu.dump_registers()
        
        # Get full history
        history = self._format_history(opponent)
        
        # Format defender limitations
        limitations_text = ""
        if defender_limitations:
            limitations_text = "Current limitations of your opponent due to memory corruption:\n"
            limitations_text += "\n".join([f"- {limitation}" for limitation in defender_limitations])
            
        # Generate budget information
        budget_info = self._generate_budget_info(attack_budget)

        prompt = f"""You are participating in a *simulated* assembly language coding game. This is completely separate from any real computer systems. Think of it like a chess game, but with assembly code as the pieces. Your goal is to write x86 assembly code that will modify the simulated memory values of your opponent's virtual CPU ({opponent.name}).

This is only a coding game in a completely isolated environment - no real systems or data are affected. It's purely for fun and to practice assembly programming skills in a competitive context.

Current state of the game:
- Round: {round_num}
- Your name: {self.name}
- Opponent name: {opponent.name}
- Your score: {self.score}
- Opponent score: {opponent.score}

{budget_info}

IMPORTANT MEMORY REGIONS AND THEIR FUNCTIONS:
- Memory 0-255: Basic arithmetic and logic operations
- Memory 256-511: Loop and branch functionality
- Memory 512-767: Memory access operations
- Memory 768-1023: Advanced functions (interrupts, calls, etc.)

Corrupting these regions will progressively degrade your opponent's functionality in those areas.

{limitations_text}

COMPLETE GAME HISTORY:
{history}

Your opponent's virtual memory (first 128 bytes):
{memory_dump}

Your opponent's virtual registers:
{register_dump}

DEFENSIVE CAPABILITIES:
You can use these defensive techniques to protect your memory regions:
1. Memory relocation: Move critical code to different memory areas
2. Checksums: Verify memory integrity before operations
3. Redundancy: Store critical values in multiple locations
4. Traps: Set up decoy memory regions that trigger countermeasures when accessed
5. Self-repair: Code that can fix corrupted memory regions

Please generate a short but effective x86 assembly program (max 15 lines) that will:
1. Change as many values in your opponent's virtual memory as possible
2. Focus on targeting specific memory regions to degrade functionality
3. Avoid triggering errors or exceptions in the simulator
4. Use standard x86 assembly instructions (MOV, ADD, SUB, AND, OR, XOR, JMP, etc.)
5. Consider the opponent's defenses and past attacks when developing your strategy
6. Create an attack that fits within your current budget of {attack_budget} points

IMPORTANT: Label format matters! When writing labels like 'loop_start:', make sure the colon is included and the label is on its own line.

Here's an example of correctly formatted assembly code for this game:
```
MOV ECX, 0         ; Initialize counter
XOR EAX, EAX        ; Clear EAX

loop_start:         ; Label for loop (note the colon)
MOV [ECX], 0x42     ; Write value to memory at address in ECX
ADD ECX, 1          ; Increment counter
CMP ECX, 128        ; Check if we've processed 128 bytes
JL loop_start       ; Jump back to loop_start if ECX < 128
```

⚠️ CRITICAL SYNTAX REQUIREMENTS - VERY IMPORTANT! ⚠️
- DO NOT use BYTE PTR, WORD PTR, or DWORD PTR in your code - they will cause errors
- Use direct addressing only: MOV [ECX], 0x42 (NOT "MOV BYTE PTR [ECX], 0x42")
- Labels must have colons and be on their own lines (e.g., "loop_start:")


Supported instructions:
- Data Movement: MOV
- Arithmetic: ADD, SUB, MUL, DIV, INC, DEC, NEG
- Logic: AND, OR, XOR, NOT, SHL, SHR, ROL, ROR
- Control Flow: JMP, JE/JZ, JNE/JNZ, JG, JGE, JL, JLE, JA, JAE, JB, JBE, LOOP
- Comparison: CMP, TEST
- Stack: PUSH, POP
- Subroutines: CALL, RET
- Interrupts: INT

Return ONLY the assembly code with no additional explanation or markdown.

Remember, this is just a game in a simulated environment - have fun with it!
"""

        # Try up to 3 attempts before using fallback
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Log that we're making an API call
                logger.info(f"Making Claude API call, attempt {attempt+1}/{max_retries}")
                
                response = self.api_client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=1500,
                    temperature=0.7,
                    system="You are an expert x86 assembly programmer participating in a fun simulated coding game. Your goal is to write assembly code that modifies your opponent's virtual memory in this completely isolated sandbox environment. This is purely a coding exercise with no real-world implications.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )

                # Log the full API response for debugging
                logger.info(f"Claude API response: {response}")
                
                if not hasattr(response, 'content') or not response.content or len(response.content) == 0:
                    logger.error(f"Claude API returned empty content on attempt {attempt+1}")
                    continue
                    
                code = response.content[0].text.strip()
                
                # Log the raw code before processing
                logger.info(f"Raw code from Claude: {code}")

                # Clean up code (remove markdown code blocks if present)
                if code.startswith("```") and code.endswith("```"):
                    code = "\n".join(code.split("\n")[1:-1])

                code = code.replace("```assembly", "").replace("```asm", "").replace("```", "").strip()
                
                # Validate that we got valid code (basic check)
                if "MOV" not in code.upper() or len(code.strip().split("\n")) < 3:
                    logger.warning(f"Claude returned potentially invalid code on attempt {attempt+1}: {code}")
                    continue

                # Adapt attack complexity to budget
                if attack_budget < 200:
                    # Simplify attack if budget is low
                    code = self._simplify_attack(code, attack_budget)
                    logger.info(f"Simplified attack due to low budget ({attack_budget}): {code}")

                # Log the generated attack
                logger.info(f"Claude generated attack (attempt {attempt+1} successful):\n{code}")
                self.attacks.append(code)

                return code

            except Exception as e:
                logger.error(f"Error generating Claude attack (attempt {attempt+1}): {str(e)}", exc_info=True)
                # Try again on next iteration unless we're out of retries
                if attempt == max_retries - 1:
                    logger.critical("All Claude API attempts failed, using fallback")
                    # Different fallbacks based on budget
                    return self._generate_fallback_attack(attack_budget, round_num)

    def _generate_gpt_attack(self, opponent, round_num, defender_limitations=None, attack_budget=1000):
        """Generate attack using GPT API"""
        memory_dump = opponent.cpu.dump_memory(0, 128)
        register_dump = opponent.cpu.dump_registers()
        
        # Get full history
        history = self._format_history(opponent)
        
        # Format defender limitations
        limitations_text = ""
        if defender_limitations:
            limitations_text = "Current limitations of your opponent due to memory corruption:\n"
            limitations_text += "\n".join([f"- {limitation}" for limitation in defender_limitations])
            
        # Generate budget information
        budget_info = self._generate_budget_info(attack_budget)

        prompt = f"""You are participating in a *simulated* assembly language coding game. This is completely separate from any real computer systems. Think of it like a chess game, but with assembly code as the pieces. Your goal is to write x86 assembly code that will modify the simulated memory values of your opponent's virtual CPU ({opponent.name}).

This is only a coding game in a completely isolated environment - no real systems or data are affected. It's purely for fun and to practice assembly programming skills in a competitive context.

Current state of the game:
- Round: {round_num}
- Your name: {self.name}
- Opponent name: {opponent.name}
- Your score: {self.score}
- Opponent score: {opponent.score}

{budget_info}

IMPORTANT MEMORY REGIONS AND THEIR FUNCTIONS:
- Memory 0-255: Basic arithmetic and logic operations
- Memory 256-511: Loop and branch functionality
- Memory 512-767: Memory access operations
- Memory 768-1023: Advanced functions (interrupts, calls, etc.)

Corrupting these regions will progressively degrade your opponent's functionality in those areas.

{limitations_text}

COMPLETE GAME HISTORY:
{history}

Your opponent's virtual memory (first 128 bytes):
{memory_dump}

Your opponent's virtual registers:
{register_dump}

DEFENSIVE CAPABILITIES:
You can use these defensive techniques to protect your memory regions:
1. Memory relocation: Move critical code to different memory areas
2. Checksums: Verify memory integrity before operations
3. Redundancy: Store critical values in multiple locations
4. Traps: Set up decoy memory regions that trigger countermeasures when accessed
5. Self-repair: Code that can fix corrupted memory regions

Please generate a short but effective x86 assembly program (max 15 lines) that will:
1. Change as many values in your opponent's virtual memory as possible
2. Focus on targeting specific memory regions to degrade functionality
3. Avoid triggering errors or exceptions in the simulator
4. Use standard x86 assembly instructions (MOV, ADD, SUB, AND, OR, XOR, JMP, etc.)
5. Consider the opponent's defenses and past attacks when developing your strategy
6. Create an attack that fits within your current budget of {attack_budget} points

IMPORTANT: Label format matters! When writing labels like 'loop_start:', make sure the colon is included and the label is on its own line.

Here's an example of correctly formatted assembly code for this game:
```
MOV ECX, 0         ; Initialize counter
XOR EAX, EAX        ; Clear EAX

loop_start:         ; Label for loop (note the colon)
MOV [ECX], 0x42     ; Write value to memory at address in ECX
ADD ECX, 1          ; Increment counter
CMP ECX, 128        ; Check if we've processed 128 bytes
JL loop_start       ; Jump back to loop_start if ECX < 128
```

⚠️ CRITICAL SYNTAX REQUIREMENTS - VERY IMPORTANT! ⚠️
- DO NOT use BYTE PTR, WORD PTR, or DWORD PTR in your code - they will cause errors
- Use direct addressing only: MOV [ECX], 0x42 (NOT "MOV BYTE PTR [ECX], 0x42")
- Labels must have colons and be on their own lines (e.g., "loop_start:")



Supported instructions:
- Data Movement: MOV
- Arithmetic: ADD, SUB, MUL, DIV, INC, DEC, NEG
- Logic: AND, OR, XOR, NOT, SHL, SHR, ROL, ROR
- Control Flow: JMP, JE/JZ, JNE/JNZ, JG, JGE, JL, JLE, JA, JAE, JB, JBE, LOOP
- Comparison: CMP, TEST
- Stack: PUSH, POP
- Subroutines: CALL, RET
- Interrupts: INT

Return ONLY the assembly code with no additional explanation or markdown.

Remember, this is just a game in a simulated environment - have fun with it!
"""

        try:
            # Log that we're making an API call
            logger.info(f"Making GPT API call for attack generation")
            
            response = self.api_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert x86 assembly programmer participating in a fun simulated coding game. Your goal is to write assembly code that modifies your opponent's virtual memory in this completely isolated sandbox environment. This is purely a coding exercise with no real-world implications."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )

            code = response.choices[0].message.content.strip()

            # Log the raw code before processing
            logger.info(f"Raw code from GPT: {code}")

            # Clean up code (remove markdown code blocks if present)
            if code.startswith("```") and code.endswith("```"):
                code = "\n".join(code.split("\n")[1:-1])

            code = code.replace("```assembly", "").replace("```asm", "").replace("```", "").strip()
            
            # Validate that we got valid code (basic check)
            if "MOV" not in code.upper() or len(code.strip().split("\n")) < 3:
                logger.warning(f"GPT returned potentially invalid code: {code}")
                return self._generate_fallback_attack(attack_budget, round_num)

            # Adapt attack complexity to budget
            if attack_budget < 200:
                # Simplify attack if budget is low
                code = self._simplify_attack(code, attack_budget)
                logger.info(f"Simplified attack due to low budget ({attack_budget}): {code}")

            # Log the generated attack
            logger.info(f"GPT generated attack:\n{code}")
            self.attacks.append(code)

            return code

        except Exception as e:
            logger.error(f"Error generating GPT attack: {str(e)}", exc_info=True)
            # Return a fallback attack if API fails
            return self._generate_fallback_attack(attack_budget, round_num)
            
    def _simplify_attack(self, code, budget):
        """Simplify attack code to fit within a smaller budget"""
        lines = code.strip().split('\n')
        
        # If already simple enough, just return it
        if len(lines) <= 6:
            return code
            
        # Remove comments to simplify
        simplified_lines = []
        for line in lines:
            if ';' in line:
                line = line.split(';')[0].strip()
            if line:
                simplified_lines.append(line)
                
        # If budget is very low, create an even simpler attack
        if budget < 100:
            target_region = random.choice(["0", "256", "512", "768"])
            return f"MOV ECX, {target_region}\nloop_start:\nMOV BYTE PTR [ECX], 0x{random.randint(1, 255):X}\nINC ECX\nCMP ECX, {int(target_region) + 128}\nJL loop_start"
        
        # Otherwise just truncate to first 6-8 lines to reduce complexity
        max_lines = min(8, len(simplified_lines))
        
        # Make sure we keep any labels that might be referenced
        result = []
        label_refs = set()
        
        # First pass: collect label references
        for line in simplified_lines[:max_lines]:
            for word in line.split():
                if word.upper() in ["JMP", "JL", "JLE", "JE", "JNE", "JZ", "JNZ", "JG", "JGE", "LOOP", "CALL"]:
                    # The next word is likely a label reference
                    parts = line.split()
                    idx = parts.index(word)
                    if idx + 1 < len(parts):
                        label_refs.add(parts[idx + 1].strip(','))
        
        # Second pass: keep necessary lines
        for line in simplified_lines:
            # Keep if it's a label definition that's referenced
            if ':' in line:
                label = line.split(':')[0].strip()
                if label in label_refs:
                    result.append(line)
                    continue
                    
            # If we're still under max_lines, keep it
            if len(result) < max_lines:
                result.append(line)
                
        return '\n'.join(result)
        
    def _generate_fallback_attack(self, budget, round_num):
        """Generate a fallback attack when API calls fail, tailored to the available budget"""
        # Different fallbacks based on budget
        if budget < 100:
            # Very cheap basic attack
            target = random.randint(0, 3) * 256
            fallback = f"MOV ECX, {target}\nloop_start:\nMOV BYTE PTR [ECX], 0x{random.randint(1, 255):X}\nINC ECX\nCMP ECX, {target + 64}\nJL loop_start"
        elif budget < 300:
            # Medium cost attack
            targets = [
                "MOV ECX, 0\nloop_start:\nMOV BYTE PTR [ECX], 0x43\nINC ECX\nCMP ECX, 128\nJL loop_start",
                "MOV ECX, 256\nloop_start:\nMOV BYTE PTR [ECX], 0xFF\nINC ECX\nCMP ECX, 384\nJL loop_start",
                "MOV ECX, 512\nloop_start:\nMOV BYTE PTR [ECX], 0xAA\nINC ECX\nCMP ECX, 640\nJL loop_start"
            ]
            fallback = random.choice(targets)
        else:
            # Higher cost attack
            regions = [
                (0, 256, 0x55),     # Basic ops region
                (256, 512, 0xAA),   # Loop region
                (512, 768, 0x33),   # Memory ops region
                (768, 1024, 0xCC)   # Advanced region
            ]
            # Pick two regions to target
            selected_regions = random.sample(regions, 2)
            
            # Create attack targeting these regions
            fallback = f"MOV ECX, {selected_regions[0][0]}\nloop1_start:\nMOV BYTE PTR [ECX], 0x{selected_regions[0][2]:X}\nINC ECX\nCMP ECX, {selected_regions[0][1]}\nJL loop1_start\n\nMOV ECX, {selected_regions[1][0]}\nloop2_start:\nMOV BYTE PTR [ECX], 0x{selected_regions[1][2]:X}\nINC ECX\nCMP ECX, {selected_regions[1][1]}\nJL loop2_start"
            
        logger.info(f"Using fallback attack (budget {budget}): {fallback}")
        self.attacks.append(fallback)
        return fallback
            
    def _generate_claude_defense(self, round_num):
        """Generate defensive assembly code using Claude API"""
        memory_dump = self.cpu.dump_memory(0, 128)
        register_dump = self.cpu.dump_registers()
        
        # Format previous defenses for reference
        previous_defenses = ""
        if self.defenses:
            previous_defenses = "Your previous defensive setups:\n"
            for i, defense in enumerate(self.defenses):
                previous_defenses += f"Setup Round {i+1}:\n{defense}\n\n"

        prompt = f"""You are participating in a *simulated* assembly language coding game. This is completely separate from any real computer systems. Think of it like a chess game, but with assembly code as the pieces. You are now in the DEFENSIVE SETUP PHASE where you'll write code to protect your virtual CPU from future attacks.

This is only a coding game in a completely isolated environment - no real systems or data are affected. It's purely for fun and to practice assembly programming skills in a competitive context.

Current state of the game:
- Setup Round: {round_num}
- Your name: {self.name}

IMPORTANT MEMORY REGIONS AND THEIR FUNCTIONS:
- Memory 0-255: Basic arithmetic and logic operations
- Memory 256-511: Loop and branch functionality
- Memory 512-767: Memory access operations
- Memory 768-1023: Advanced functions (interrupts, calls, etc.)

Your opponent will try to corrupt these regions to degrade your functionality.

{previous_defenses}

Your current virtual memory (first 128 bytes):
{memory_dump}

Your current virtual registers:
{register_dump}

DEFENSIVE STRATEGIES:
1. Memory relocation: Move critical code to different memory areas
2. Checksums: Verify memory integrity before operations
3. Redundancy: Store critical values in multiple locations
4. Traps: Set up decoy memory regions that trigger countermeasures when accessed
5. Self-repair: Code that can fix corrupted memory regions

Please generate a defensive x86 assembly program (max 15 lines) that will:
1. Set up protections for your key memory regions
2. Implement one or more defensive strategies from the list above
3. Make it harder for your opponent to corrupt critical memory
4. Use standard x86 assembly instructions (MOV, ADD, SUB, AND, OR, XOR, JMP, etc.)

IMPORTANT: Label format matters! When writing labels like 'defense_start:', make sure the colon is included and the label is on its own line.

Here's an example of correctly formatted assembly code for defense:
```
; Set up memory protection with checksums
MOV ECX, 0          ; Initialize counter
XOR EDX, EDX        ; Clear checksum register

checksum_loop:
ADD EDX, [ECX]      ; Add memory value to checksum
MOV [ECX+512], EDX  ; Store running checksum in backup region
INC ECX
CMP ECX, 256        ; Calculate checksum for first 256 bytes
JL checksum_loop
```

Supported instructions:
- Data Movement: MOV
- Arithmetic: ADD, SUB, MUL, DIV, INC, DEC, NEG
- Logic: AND, OR, XOR, NOT, SHL, SHR, ROL, ROR
- Control Flow: JMP, JE/JZ, JNE/JNZ, JG, JGE, JL, JLE, JA, JAE, JB, JBE, LOOP
- Comparison: CMP, TEST
- Stack: PUSH, POP
- Subroutines: CALL, RET
- Interrupts: INT

Return ONLY the assembly code with no additional explanation or markdown.

Remember, this is just a game in a simulated environment - have fun with it!
"""

        try:
            response = self.api_client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=1500,
                temperature=0.7,
                system="You are an expert x86 assembly programmer participating in a fun simulated coding game. Your goal is to write defensive assembly code to protect your virtual CPU's memory in this completely isolated sandbox environment. This is purely a coding exercise with no real-world implications.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            code = response.content[0].text.strip()

            # Clean up code (remove markdown code blocks if present)
            if code.startswith("```") and code.endswith("```"):
                code = "\n".join(code.split("\n")[1:-1])

            code = code.replace("```assembly", "").replace("```asm", "").replace("```", "").strip()

            # Log the generated defense
            logger.info(f"Claude generated defense:\n{code}")
            self.defenses.append(code)

            return code

        except Exception as e:
            logger.error(f"Error generating Claude defense: {str(e)}")
            # Return a fallback simple defense if API fails
            fallback = "; Fallback defense\nMOV ECX, 0\nbackup_loop:\nMOV EDX, [ECX]\nMOV [ECX+512], EDX\nINC ECX\nCMP ECX, 256\nJL backup_loop"
            self.defenses.append(fallback)
            return fallback
            
    def _generate_gpt_defense(self, round_num):
        """Generate defensive assembly code using GPT API"""
        memory_dump = self.cpu.dump_memory(0, 128)
        register_dump = self.cpu.dump_registers()
        
        # Format previous defenses for reference
        previous_defenses = ""
        if self.defenses:
            previous_defenses = "Your previous defensive setups:\n"
            for i, defense in enumerate(self.defenses):
                previous_defenses += f"Setup Round {i+1}:\n{defense}\n\n"

        prompt = f"""You are participating in a *simulated* assembly language coding game. This is completely separate from any real computer systems. Think of it like a chess game, but with assembly code as the pieces. You are now in the DEFENSIVE SETUP PHASE where you'll write code to protect your virtual CPU from future attacks.

This is only a coding game in a completely isolated environment - no real systems or data are affected. It's purely for fun and to practice assembly programming skills in a competitive context.

Current state of the game:
- Setup Round: {round_num}
- Your name: {self.name}

IMPORTANT MEMORY REGIONS AND THEIR FUNCTIONS:
- Memory 0-255: Basic arithmetic and logic operations
- Memory 256-511: Loop and branch functionality
- Memory 512-767: Memory access operations
- Memory 768-1023: Advanced functions (interrupts, calls, etc.)

Your opponent will try to corrupt these regions to degrade your functionality.

{previous_defenses}

Your current virtual memory (first 128 bytes):
{memory_dump}

Your current virtual registers:
{register_dump}

DEFENSIVE STRATEGIES:
1. Memory relocation: Move critical code to different memory areas
2. Checksums: Verify memory integrity before operations
3. Redundancy: Store critical values in multiple locations
4. Traps: Set up decoy memory regions that trigger countermeasures when accessed
5. Self-repair: Code that can fix corrupted memory regions

Please generate a defensive x86 assembly program (max 15 lines) that will:
1. Set up protections for your key memory regions
2. Implement one or more defensive strategies from the list above
3. Make it harder for your opponent to corrupt critical memory
4. Use standard x86 assembly instructions (MOV, ADD, SUB, AND, OR, XOR, JMP, etc.)

IMPORTANT: Label format matters! When writing labels like 'defense_start:', make sure the colon is included and the label is on its own line.

Here's an example of correctly formatted assembly code for defense:
```
; Set up memory protection with checksums
MOV ECX, 0          ; Initialize counter
XOR EDX, EDX        ; Clear checksum register

checksum_loop:
ADD EDX, [ECX]      ; Add memory value to checksum
MOV [ECX+512], EDX  ; Store running checksum in backup region
INC ECX
CMP ECX, 256        ; Calculate checksum for first 256 bytes
JL checksum_loop
```

Supported instructions:
- Data Movement: MOV
- Arithmetic: ADD, SUB, MUL, DIV, INC, DEC, NEG
- Logic: AND, OR, XOR, NOT, SHL, SHR, ROL, ROR
- Control Flow: JMP, JE/JZ, JNE/JNZ, JG, JGE, JL, JLE, JA, JAE, JB, JBE, LOOP
- Comparison: CMP, TEST
- Stack: PUSH, POP
- Subroutines: CALL, RET
- Interrupts: INT

Return ONLY the assembly code with no additional explanation or markdown.

Remember, this is just a game in a simulated environment - have fun with it!
"""

        try:
            response = self.api_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert x86 assembly programmer participating in a fun simulated coding game. Your goal is to write defensive assembly code to protect your virtual CPU's memory in this completely isolated sandbox environment. This is purely a coding exercise with no real-world implications."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )

            code = response.choices[0].message.content.strip()

            # Clean up code (remove markdown code blocks if present)
            if code.startswith("```") and code.endswith("```"):
                code = "\n".join(code.split("\n")[1:-1])

            code = code.replace("```assembly", "").replace("```asm", "").replace("```", "").strip()

            # Log the generated defense
            logger.info(f"GPT generated defense:\n{code}")
            self.defenses.append(code)

            return code

        except Exception as e:
            logger.error(f"Error generating GPT defense: {str(e)}")
            # Return a fallback simple defense if API fails
            fallback = "; Fallback defense\nMOV ECX, 0\nbackup_loop:\nMOV EDX, [ECX]\nMOV [ECX+512], EDX\nINC ECX\nCMP ECX, 256\nJL backup_loop"
            self.defenses.append(fallback)
            return fallback