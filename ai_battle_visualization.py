from manim import *
import json
import numpy as np
import re
import random
import math
config.background_color = BLACK

class MemoryRegion(Rectangle):
    def __init__(self, name, start, length, color=BLUE, width=10, height=1.5, **kwargs):
        super().__init__(width=width, height=height, fill_opacity=0.7, stroke_width=3, **kwargs)
        self.name = name
        self.start = start
        self.length = length
        self.set_fill(color)
        self.set_stroke(WHITE)

        # Create label
        self.label = Text(f"{name} ({start}-{start+length-1})", font_size=20)
        self.label.next_to(self, UP, buff=0.1)

        # Create health bar
        self.health = 100
        self.health_bar_bg = Rectangle(width=width-0.5, height=0.2, fill_opacity=1, color=GREY)
        self.health_bar_bg.next_to(self, DOWN, buff=0.1)
        self.health_bar = Rectangle(width=width-0.5, height=0.2, fill_opacity=1, color=GREEN)
        self.health_bar.next_to(self, DOWN, buff=0.1)

        # Create memory cells
        self.cells = VGroup()
        cols = 16
        rows = math.ceil(length / cols)
        cell_width = (width - 0.4) / cols
        cell_height = (height - 0.4) / rows

        for i in range(length):
            row = i // cols
            col = i % cols
            x = -width/2 + 0.2 + col * cell_width + cell_width/2
            y = height/2 - 0.2 - row * cell_height - cell_height/2
            cell = Square(side_length=min(cell_width, cell_height)*0.8, fill_opacity=0.2, color=WHITE)
            cell.move_to([x, y, 0])
            self.cells.add(cell)

    def get_region_with_label(self):
        return VGroup(self, self.label)

    def get_full_region(self):
        return VGroup(self, self.label, self.health_bar_bg, self.health_bar, self.cells)

    def update_health(self, damage_percent):
        self.health = max(0, self.health - damage_percent)
        health_color = interpolate_color(RED, GREEN, self.health/100)
        new_width = (self.width-0.5) * (self.health/100)

        self.health_bar = Rectangle(width=new_width, height=0.2, fill_opacity=1, color=health_color)
        self.health_bar.align_to(self.health_bar_bg, LEFT)
        self.health_bar.set_y(self.health_bar_bg.get_y())

        return self.health_bar

    def corrupt_cells(self, indices):
        corrupted = VGroup()
        for idx in indices:
            if 0 <= idx < len(self.cells):
                cell = self.cells[idx]
                corrupted.add(cell)
        return corrupted

class AssemblyCode(VGroup):
    def __init__(self, code, **kwargs):
        super().__init__(**kwargs)
        self.code_text = code
        self.code_lines = code.strip().split('\n')

        # Apply syntax highlighting
        self.lines = VGroup()
        for i, line in enumerate(self.code_lines[:12]):  # Limit to 12 lines
            highlighted_line = self.highlight_syntax(line)
            highlighted_line.shift(i * DOWN * 0.4)
            self.lines.add(highlighted_line)

        self.add(self.lines)
        self.scale(0.5)  # Scale down to fit

    def highlight_syntax(self, line):
        line_parts = VGroup()
        x_pos = 0

        # Handle comments
        if ';' in line:
            code_part, comment_part = line.split(';', 1)
            comment = Text(f"; {comment_part}", color=GREEN, font_size=24)
        else:
            code_part, comment_part = line, ""
            comment = None

        # Split by known tokens for highlighting
        parts = re.split(r'(\bMOV\b|\bXOR\b|\bADD\b|\bINC\b|\bDEC\b|\bJMP\b|\bJNZ\b|\bJLE?\b|\bLOOP\b|\bCMP\b|\bESI\b|\bEDI\b|\bE[ABCD]X\b|\[.*?\]|0x[0-9A-Fa-f]+)', code_part)

        for part in parts:
            if not part:
                continue

            if re.match(r'\b(MOV|XOR|ADD|INC|DEC|JMP|JNZ|JL|JLE|LOOP|CMP)\b', part, re.IGNORECASE):
                # Instructions in cyan
                text = Text(part, color=BLUE_C, font_size=24)
            elif re.match(r'\b(ESI|EDI|E[ABCD]X)\b', part, re.IGNORECASE):
                # Registers in orange
                text = Text(part, color=GOLD, font_size=24)
            elif re.match(r'\[.*?\]', part):
                # Memory references in purple
                text = Text(part, color=PURPLE, font_size=24)
            elif re.match(r'0x[0-9A-Fa-f]+', part):
                # Hex values in yellow
                text = Text(part, color=YELLOW, font_size=24)
            else:
                # Other text in white
                text = Text(part, color=WHITE, font_size=24)

            text.next_to(line_parts, RIGHT, buff=0.05) if line_parts else text.move_to([x_pos, 0, 0], aligned_edge=LEFT)
            line_parts.add(text)
            x_pos += text.width

        # Add comment at the end if it exists
        if comment:
            comment.next_to(line_parts, RIGHT, buff=0.3) if line_parts else comment.move_to([x_pos, 0, 0], aligned_edge=LEFT)
            line_parts.add(comment)

        return line_parts

class AttackArrow(Arrow):
    def __init__(self, attacker, **kwargs):
        color = BLUE if attacker == "Claude" else GREEN
        super().__init__(start=2*LEFT, end=2*RIGHT, buff=0.1, color=color, **kwargs)
        self.attacker = attacker

        # Add label
        self.label = Text(f"{attacker} Attack", color=color, font_size=24)
        self.label.next_to(self, UP, buff=0.1)

class BattleVisualization(Scene):
    def construct(self):
        # Load the battle data
        with open('battle_history.json', 'r') as f:
            self.battle_data = json.load(f)

        # Load the battle log
        with open('battle_log.txt', 'r') as f:
            self.battle_log = f.read()

        # Parse the battle log
        self.rounds = self.parse_battle_log()

        # Set up memory regions
        self.setup_memory_regions()

        # Create title
        title = Text("AI Assembly Code Battle", font_size=48, color=WHITE)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))

        # Process each round
        for round_num, round_data in enumerate(self.rounds[:10]):  # Limit to first 10 rounds for demo
            self.visualize_round(round_num, round_data)

        # Show final results
        self.show_final_results()

    def parse_battle_log(self):
        rounds = []
        current_round = {}
        attack_code = []
        in_attack_code = False
        execution_results = {}

        for line in self.battle_log.split('\n'):
            line = line.strip()

            # New round detection
            if "ROUND" in line and "==============" in line:
                round_match = re.search(r'ROUND (\d+)', line)
                if round_match:
                    if current_round and attack_code:
                        current_round["attack_code"] = "\n".join(attack_code)
                        current_round.update(execution_results)
                        rounds.append(current_round)

                    current_round = {"round": int(round_match.group(1))}
                    attack_code = []
                    in_attack_code = False
                    execution_results = {}

            # Attacker/Defender detection
            elif "attacks" in line and not in_attack_code:
                attacker_match = re.search(r'(\w+) attacks (\w+)', line)
                if attacker_match:
                    current_round["attacker"] = attacker_match.group(1)
                    current_round["defender"] = attacker_match.group(2)

            # Attack code section
            elif "Attack Code:" in line:
                in_attack_code = True
            elif "Execution Result:" in line:
                in_attack_code = False
            elif in_attack_code:
                attack_code.append(line)

            # Result information
            elif "Instructions executed:" in line:
                val = re.search(r'Instructions executed: (\d+)', line)
                if val:
                    execution_results["instructions"] = int(val.group(1))
            elif "Memory locations corrupted:" in line:
                val = re.search(r'Memory locations corrupted: (\d+)', line)
                if val:
                    execution_results["corrupted"] = int(val.group(1))
            elif "Error:" in line:
                val = re.search(r'Error: (.+)', line)
                if val:
                    execution_results["error"] = val.group(1)
            elif "Attack cost:" in line:
                val = re.search(r'Attack cost: (\d+)', line)
                if val:
                    execution_results["cost"] = int(val.group(1))
            elif "Remaining budget:" in line:
                val = re.search(r'Remaining budget: (\d+)', line)
                if val:
                    execution_results["remaining_budget"] = int(val.group(1))

        # Add the last round
        if current_round and attack_code:
            current_round["attack_code"] = "\n".join(attack_code)
            current_round.update(execution_results)
            rounds.append(current_round)

        return rounds

    def setup_memory_regions(self):
        # Define the memory regions
        self.memory_regions = {
            "basic_ops": MemoryRegion("Basic Operations", 0, 128, color=BLUE_B),
            "loops": MemoryRegion("Loops/Branching", 256, 256, color=GREEN_B),
            "memory_ops": MemoryRegion("Memory Access", 512, 256, color=RED_B),
            "advanced_ops": MemoryRegion("Advanced Functions", 768, 256, color=YELLOW_B)
        }

        # Position the memory regions
        memory_group = VGroup()
        y_offset = 0

        for i, (name, region) in enumerate(self.memory_regions.items()):
            full_region = region.get_full_region()
            if i == 0:
                full_region.move_to(ORIGIN)
            else:
                full_region.next_to(memory_group, DOWN, buff=0.5)
            memory_group.add(full_region)

        memory_group.scale(0.8)
        memory_group.move_to(ORIGIN)

        # Create and display the memory layout
        self.play(
            *[Create(region) for region in self.memory_regions.values()],
            *[Write(region.label) for region in self.memory_regions.values()],
            *[Create(region.health_bar_bg) for region in self.memory_regions.values()],
            *[Create(region.health_bar) for region in self.memory_regions.values()],
            run_time=2
        )

        # Optionally show memory cells
        self.play(
            *[FadeIn(region.cells) for region in self.memory_regions.values()],
            run_time=1
        )

    def visualize_round(self, round_idx, round_data):
        round_num = round_data["round"]
        attacker = round_data["attacker"]
        defender = round_data["defender"]
        attack_code = round_data["attack_code"]
        corrupted = round_data.get("corrupted", 0)
        cost = round_data.get("cost", 0)
        error = round_data.get("error", "None")

        # Show round header
        round_header = Text(f"Round {round_num}: {attacker} attacks {defender}!",
                           font_size=36,
                           color=BLUE if attacker == "Claude" else GREEN)
        round_header.to_edge(UP, buff=1.5)
        self.play(Write(round_header))

        # Show attack code
        code_box = Rectangle(width=5.5, height=4.5, fill_opacity=0.2, color=GREY)  # Reduced from 6x5
        code_box.to_edge(RIGHT, buff=1.5)  # Increased buffer
        code_title = Text("Attack Code", font_size=30)
        code_title.next_to(code_box, UP, buff=0.2)

        code = AssemblyCode(attack_code)
        code.scale(0.8)  # Add scaling to fit better
        code.move_to(code_box.get_center())
    
        self.play(
            Create(code_box),
            Write(code_title),
            Write(code)
        )

        # Create attack animation
        attack_arrow = Arrow(
            start=code_box.get_left(),  # Start at code box
            end=memory_group.get_right(), # End at memory group
            buff=0.2,  # Increased buffer
            color=attacker_color
        )

        self.play(
            Write(attack_arrow.label),
            GrowArrow(attack_arrow)
        )

        # Animate attack particles
        num_particles = min(corrupted, 50)  # Limit particles to keep animation smooth
        particles = VGroup()

        for _ in range(num_particles):
            dot = Dot(color=BLUE if attacker == "Claude" else GREEN, radius=0.05)
            dot.move_to(attack_arrow.get_start())
            particles.add(dot)

        # Determine which memory regions to attack based on who's attacking
        target_memory = [region for region in self.memory_regions.values()]

        # Create separate animations for each particle
        particle_animations = []
        corrupted_cells = []

        for i, particle in enumerate(particles):
            # Pick a random region and a random cell in that region
            target_region = random.choice(target_memory)
            target_cell_idx = random.randint(0, target_region.length - 1)
            target_cell = target_region.cells[target_cell_idx]

            anim = MoveAlongPath(
                particle,
                CubicBezier(
                    attack_arrow.get_start(),
                    attack_arrow.get_start() + UP*random.uniform(-1, 1) + RIGHT*random.uniform(0, 1),
                    target_cell.get_center() + UP*random.uniform(-1, 1) + LEFT*random.uniform(0, 1),
                    target_cell.get_center()
                ),
                rate_func=rate_functions.ease_in_out_quad,
                run_time=random.uniform(0.5, 1.5)
            )
            particle_animations.append(anim)
            corrupted_cells.append((target_region, target_cell_idx))

        # Play particle animations
        self.play(
            *particle_animations,
            run_time=2
        )

        # Corrupt memory cells and update health bars
        health_bar_animations = []
        corrupt_animations = []

        for region, cell_idx in corrupted_cells:
            cell = region.cells[cell_idx]
            corrupt_animations.append(cell.animate.set_fill(RED, opacity=0.8))

            # Update health (based on number of cells corrupted for this region)
            region_corrupted = sum(1 for r, _ in corrupted_cells if r == region)
            damage = min(100, region_corrupted / region.length * 100)
            new_health_bar = region.update_health(damage/len(corrupted_cells))
            health_bar_animations.append(Transform(region.health_bar, new_health_bar))

        self.play(
            *corrupt_animations,
            *health_bar_animations,
            run_time=1
        )

        # Show attack results
        results_box = Rectangle(width=5.5, height=1.8, fill_opacity=0.2, color=GREY)  # Smaller
        results_box.next_to(code_box, DOWN, buff=0.3)  # Closer to code box
        results_title = Text("Attack Results", font_size=30)
        results_title.next_to(results_box, UP, buff=0.2)

        results_text = VGroup(
            Text(f"Memory corrupted: {corrupted} locations", font_size=24),
            Text(f"Attack cost: {cost} points", font_size=24),
            Text(f"Error: {error[:30]}{'...' if len(error) > 30 else ''}", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        results_text.move_to(results_box.get_center())

        self.play(
            Create(results_box),
            Write(results_title),
            Write(results_text)
        )

        # Wait and clean up for next round
        self.wait(1)
        self.play(
            FadeOut(round_header),
            FadeOut(code_box),
            FadeOut(code_title),
            FadeOut(code),
            FadeOut(attack_arrow),
            FadeOut(attack_arrow.label),
            FadeOut(results_box),
            FadeOut(results_title),
            FadeOut(results_text),
            FadeOut(particles)
        )

    def show_final_results(self):
        # Clear everything except memory regions
        self.clear()

        # Create title
        final_title = Text("Battle Results", font_size=48, color=WHITE)
        final_title.to_edge(UP, buff=0.5)

        # Get winner and scores
        winner = self.battle_data.get("winner", "Unknown")
        claude_score = self.battle_data.get("claude_score", 0)
        gpt_score = self.battle_data.get("gpt_score", 0)
        claude_budget = self.battle_data.get("claude_final_budget", 0)
        gpt_budget = self.battle_data.get("gpt_final_budget", 0)

        # Create final text
        winner_text = Text(f"Winner: {winner}!", font_size=42, color=BLUE if winner == "Claude" else GREEN)

        score_group = VGroup(
            Text(f"Claude Score: {claude_score}", font_size=36, color=BLUE),
            Text(f"GPT Score: {gpt_score}", font_size=36, color=GREEN),
            Text(f"Claude Final Budget: {claude_budget}", font_size=30, color=BLUE_B),
            Text(f"GPT Final Budget: {gpt_budget}", font_size=30, color=GREEN_B)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)

        winner_text.to_edge(UP, buff=1.5)
        score_group.next_to(winner_text, DOWN, buff=0.5)

        # Create memory impact visualization
        impact_title = Text("Memory Region Impact", font_size=36)

        # Create bar charts for memory impact
        claude_impact = self.battle_data.get("claude_memory_impact", {})
        gpt_impact = self.battle_data.get("gpt_memory_impact", {})

        # Create axes for Claude's impact
        claude_chart = BarChart(
            values=[
                claude_impact.get("basic_ops", 0),
                claude_impact.get("loops", 0),
                claude_impact.get("memory_ops", 0),
                claude_impact.get("advanced_ops", 0)
            ],
            bar_names=["Basic Ops", "Loops", "Memory Ops", "Advanced"],
            y_range=[0, 1500, 300],
            x_length=10,
            y_length=5,
            bar_colors=[BLUE, GREEN, RED, YELLOW]
        )

        # Create axes for GPT's impact
        gpt_chart = BarChart(
            values=[
                gpt_impact.get("basic_ops", 0),
                gpt_impact.get("loops", 0),
                gpt_impact.get("memory_ops", 0),
                gpt_impact.get("advanced_ops", 0)
            ],
            bar_names=["Basic Ops", "Loops", "Memory Ops", "Advanced"],
            y_range=[0, 1500, 300],
            x_length=10,
            y_length=5,
            bar_colors=[BLUE, GREEN, RED, YELLOW]
        )

        # Position charts
        impact_title.next_to(score_group, DOWN, buff=1)

        claude_label = Text("Claude's Impact on GPT Memory", font_size=30, color=BLUE)
        claude_label.next_to(impact_title, DOWN, buff=0.3)
        claude_chart.next_to(claude_label, DOWN, buff=0.5)

        gpt_label = Text("GPT's Impact on Claude Memory", font_size=30, color=GREEN)
        gpt_label.next_to(claude_chart, DOWN, buff=0.5)
        gpt_chart.next_to(gpt_label, DOWN, buff=0.5)

        # Animate final results
        self.play(Write(final_title))
        self.play(Write(winner_text))
        self.play(Write(score_group))

        self.wait(1)

        self.play(
            Write(impact_title),
            Write(claude_label)
        )
        self.play(Create(claude_chart))

        self.wait(1)

        self.play(
            Write(gpt_label)
        )
        self.play(Create(gpt_chart))

        # Final wait
        self.wait(3)

class BattleOverview(Scene):
    def construct(self):
        # Load the battle data
        with open('battle_history.json', 'r') as f:
            self.battle_data = json.load(f)

        # Create title
        title = Text("AI Assembly Code Battle - Overview", font_size=48, color=WHITE)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))

        # Create memory map visualization
        memory_map = self.create_memory_map()
        self.play(FadeIn(memory_map))

        # Explain battle context
        battle_info = VGroup(
            Text("Claude vs GPT Assembly Code Battle", font_size=36),
            Text("Goal: Corrupt opponent's memory while protecting your own", font_size=30),
            Text("Scoring based on memory corruption and attack costs", font_size=30)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        battle_info.next_to(memory_map, DOWN, buff=0.5)

        self.play(Write(battle_info))
        self.wait(2)

        # Show battle structure
        self.play(FadeOut(battle_info))

        memory_types = VGroup(
            Text("Memory Regions:", font_size=30, color=WHITE),
            Text("Basic Operations (0-127)", font_size=24, color=BLUE),
            Text("Loops & Branching (256-511)", font_size=24, color=GREEN),
            Text("Memory Access (512-767)", font_size=24, color=RED),
            Text("Advanced Functions (768-1023)", font_size=24, color=YELLOW)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        memory_types.next_to(memory_map, DOWN, buff=0.5)

        self.play(Write(memory_types))
        self.wait(2)

        # Animate battle flow
        flow_title = Text("Battle Flow", font_size=36)
        flow_title.next_to(memory_types, DOWN, buff=0.5)

        flow_steps = VGroup(
            Text("1. Alternating attacks between Claude and GPT", font_size=24),
            Text("2. Each attack targets specific memory regions", font_size=24),
            Text("3. Damage calculated by memory locations corrupted", font_size=24),
            Text("4. Winner determined by total score and memory impact", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        flow_steps.next_to(flow_title, DOWN, buff=0.3)

        self.play(Write(flow_title))
        self.play(Write(flow_steps))

        # Show the final results preview
        self.wait(2)
        self.play(
            FadeOut(memory_map),
            FadeOut(memory_types),
            FadeOut(flow_title),
            FadeOut(flow_steps)
        )

        winner = self.battle_data.get("winner", "Unknown")
        final_text = Text(f"Final Winner: {winner}", font_size=42, color=BLUE if winner == "Claude" else GREEN)
        final_text.next_to(title, DOWN, buff=1)

        self.play(Write(final_text))
        self.wait(2)

    def create_memory_map(self):
        # Create a visual representation of the memory layout
        memory_map = VGroup()

        # Background rectangle for full memory space
        full_memory = Rectangle(width=12, height=1.5, fill_opacity=0.3, color=GREY)
        memory_map.add(full_memory)

        # Create regions
        regions = [
            {"name": "Basic Ops", "start": 0, "length": 128, "color": BLUE},
            {"name": "Loops/Branch", "start": 256, "length": 256, "color": GREEN},
            {"name": "Memory Access", "start": 512, "length": 256, "color": RED},
            {"name": "Advanced Funcs", "start": 768, "length": 256, "color": YELLOW}
        ]

        total_memory = 1024
        width = full_memory.width

        for region in regions:
            region_width = (region["length"] / total_memory) * width
            region_rect = Rectangle(
                width=region_width,
                height=1.5,
                fill_opacity=0.7,
                color=region["color"]
            )

            # Position based on start address
            x_offset = -width/2 + (region["start"] / total_memory) * width + region_width/2
            region_rect.move_to([x_offset, 0, 0])

            # Add label
            label = Text(region["name"], font_size=20, color=WHITE)
            label.next_to(region_rect, UP, buff=0.1)

            memory_map.add(region_rect, label)

        memory_map.move_to(ORIGIN)
        return memory_map

class RoundVisualization(Scene):
    def construct(self):
        # This class can be used to visualize a single round in detail
        with open('battle_history.json', 'r') as f:
            battle_data = json.load(f)

        with open('battle_log.txt', 'r') as f:
            battle_log = f.read()

        # Extract a specific round for detailed visualization
        round_text = [r for r in battle_log.split("==============") if "ROUND 1" in r][0]

        # Parse round details
        attacker = "Claude"
        defender = "GPT"
        attack_code = "\n".join([
            line for line in round_text.split("\n")
            if line.strip() and not line.startswith("[") and not "Attack Code" in line and not "Execution Result" in line
            and not "------------" in line and not "ROUND" in line and not "attacks" in line
        ][:15])  # Limit to 15 lines for cleaner visualization

        # Create title and setup
        title = Text(f"Round 1: {attacker} attacks {defender}", font_size=36, color=BLUE)
        title.to_edge(UP, buff=0.5)

        self.play(Write(title))

        # Create code display
        code = AssemblyCode(attack_code)
        code.scale(1.2)  # Make it larger for this focused visualization
        code.to_edge(LEFT, buff=1)

        code_title = Text("Attack Code", font_size=30)
        code_title.next_to(code, UP, buff=0.3)

        self.play(
            Write(code_title),
            Write(code)
        )

        # Create target memory visualization
        target_rect = Rectangle(width=6, height=5, fill_opacity=0.3, color=GREY)
        target_rect.to_edge(RIGHT, buff=1)

        target_title = Text(f"{defender}'s Memory", font_size=30, color=GREEN)
        target_title.next_to(target_rect, UP, buff=0.3)

        # Create memory regions inside target
        regions = [
            {"name": "Basic Ops", "height": 1, "color": BLUE},
            {"name": "Loops", "height": 1.5, "color": GREEN},
            {"name": "Memory Access", "height": 1.5, "color": RED},
            {"name": "Advanced", "height": 1, "color": YELLOW}
        ]

        memory_regions = VGroup()

        y_pos = target_rect.get_top()[1] - 0.5
        for region in regions:
            rect = Rectangle(
                width=5,
                height=region["height"],
                fill_opacity=0.7,
                color=region["color"]
            )
            rect.move_to([target_rect.get_center()[0], y_pos - region["height"]/2, 0])

            label = Text(region["name"], font_size=20)
            label.move_to(rect.get_center())

            memory_regions.add(rect, label)
            y_pos -= region["height"]

        self.play(
            Create(target_rect),
            Write(target_title),
            *[Create(memory_regions[i]) for i in range(len(memory_regions))],
            *[Write(memory_regions[i+1]) for i in range(0, len(memory_regions), 2)]
        )

        # Create attack animation
        attack_arrow = Arrow(code.get_right() + RIGHT*0.5, target_rect.get_left() + LEFT*0.5, buff=0.1, color=BLUE)
        attack_label = Text("Attack", font_size=24, color=BLUE)
        attack_label.next_to(attack_arrow, UP, buff=0.1)

        self.play(
            GrowArrow(attack_arrow),
            Write(attack_label)
        )

        # Create particle effect for attack
        particles = VGroup()
        for _ in range(30):
            particle = Dot(radius=0.05, color=BLUE)
            particle.move_to(code.get_right() + RIGHT*0.5)
            particles.add(particle)

        # Animate particles hitting the memory regions
        particle_animations = []

        for i, particle in enumerate(particles):
            # Target different memory regions based on particle index
            target_idx = i % 4
            target_region = memory_regions[target_idx*2]  # Get the rectangle, not the label

            # Create a path with a slight curve
            start = particle.get_center()
            end = target_region.get_center() + np.array([random.uniform(-1, 1), random.uniform(-0.5, 0.5), 0])

            control1 = start + np.array([1, random.uniform(-1, 1), 0])
            control2 = end + np.array([-1, random.uniform(-1, 1), 0])

            path = CubicBezier(start, control1, control2, end)

            # Create the animation
            anim = MoveAlongPath(particle, path, run_time=random.uniform(1, 2))
            particle_animations.append(anim)

        self.play(
            *particle_animations,
            run_time=2
        )

        # Show memory corruption effect
        corruption_animations = []

        for i, region in enumerate(regions):
            if i % 2 == 0:  # Corrupt every other region more heavily
                for _ in range(5):
                    x = random.uniform(-2, 2)
                    y = random.uniform(-region["height"]/2, region["height"]/2)

                    corruption = Dot(radius=0.1, color=RED, fill_opacity=0.8)
                    corruption.move_to(memory_regions[i*2].get_center() + np.array([x, y, 0]))

                    corruption_animations.append(FadeIn(corruption, rate_func=there_and_back_with_pause, run_time=1))

        self.play(
            *corruption_animations,
            run_time=2
        )

        # Show attack results
        results_box = Rectangle(width=6, height=2, fill_opacity=0.2, color=GREY)
        results_box.to_edge(DOWN, buff=1)

        results_title = Text("Attack Results", font_size=30)
        results_title.next_to(results_box, UP, buff=0.2)

        results = VGroup(
            Text("Memory corrupted: 604 locations", font_size=24),
            Text("Attack cost: 505 points", font_size=24),
            Text("Error: Maximum instruction limit reached", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        results.scale(0.8)
        results.move_to(results_box.get_center())

        self.play(
            Create(results_box),
            Write(results_title),
            Write(results)
        )

        self.wait(2)

# This is the main visualization class that should be rendered
class AIBattleSimulation(Scene):
    def construct(self):
        # Load battle data
        with open('battle_history.json', 'r') as f:
            battle_data = json.load(f)

        with open('battle_log.txt', 'r') as f:
            battle_log = f.read()
        self.camera.frame_width = 32
        self.camera.frame_height = 18
        # Create intro animation
        self.intro_animation()

        # Show memory structure
        self.memory_structure()

        # Process battle rounds
        self.battle_rounds(battle_data, battle_log)
        # Show final results
        self.final_results(battle_data)

    def intro_animation(self):
        # Create title
        title = Text("AI Assembly Code Battle", font_size=48)
        subtitle = Text("Claude vs GPT", font_size=36)
        subtitle.next_to(title, DOWN, buff=0.5)

        # Add CPU circuits background
        circuits = self.create_circuit_background()

        self.play(
            Write(title),
            FadeIn(circuits)
        )
        self.play(Write(subtitle))

        # Explanation text
        explanation = Text(
            "Two AI systems compete in a battle of assembly code.",
            font_size=30
        )
        explanation.next_to(subtitle, DOWN, buff=1)

        self.play(Write(explanation))
        self.wait(2)

        # Show battle format
        battle_format = VGroup(
            Text("• Each AI takes turns attacking the other's memory", font_size=28),
            Text("• Attacks use real assembly code instructions", font_size=28),
            Text("• Goal: Corrupt opponent's memory and disrupt operations", font_size=28),
            Text("• Winner determined by memory corruption and attack efficiency", font_size=28)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        battle_format.next_to(explanation, DOWN, buff=0.5)

        for line in battle_format:
            self.play(Write(line))
            self.wait(0.5)

        self.wait(2)

        # Transition out
        self.play(
            FadeOut(title),
            FadeOut(subtitle),
            FadeOut(explanation),
            FadeOut(battle_format),
            FadeOut(circuits)
        )

    def create_circuit_background(self):
        # Create a CPU circuit-like background
        circuits = VGroup()

        # Add horizontal and vertical lines
        for i in range(-5, 6, 2):
            # Horizontal lines
            h_line = Line(LEFT*6 + UP*i, RIGHT*6 + UP*i, stroke_width=1, color=BLUE_E)
            circuits.add(h_line)

            # Vertical lines
            v_line = Line(LEFT*i + UP*3, LEFT*i + DOWN*3, stroke_width=1, color=BLUE_E)
            circuits.add(v_line)

        # Add dots at intersections
        for i in range(-5, 6, 2):
            for j in range(-3, 4, 2):
                dot = Dot(point=np.array([i, j, 0]), radius=0.05, color=GREEN_E)
                circuits.add(dot)

        # Add some more circuit elements
        for _ in range(20):
            x = random.randint(-5, 5)
            y = random.randint(-3, 3)

            if random.choice([True, False]):
                # Small square
                square = Square(side_length=0.2, color=YELLOW_E, fill_opacity=0.5)
                square.move_to([x, y, 0])
                circuits.add(square)
            else:
                # Small circuit
                circle = Circle(radius=0.1, color=RED_E, fill_opacity=0.5)
                circle.move_to([x, y, 0])
                circuits.add(circle)

        circuits.set_opacity(0.3)
        return circuits

    def memory_structure(self):
        # Create memory map title
        title = Text("Memory Structure", font_size=36)
        title.to_edge(UP, buff=0.5)

        self.play(Write(title))

        # Create memory visualization
        memory_bg = Rectangle(width=10, height=1.5, fill_opacity=0.2, color=GREY)

        # Memory regions
        regions = [
            {"name": "Basic Operations", "start": 0, "length": 128, "color": BLUE},
            {"name": "Loops & Branching", "start": 256, "length": 256, "color": GREEN},
            {"name": "Memory Access", "start": 512, "length": 256, "color": RED},
            {"name": "Advanced Functions", "start": 768, "length": 256, "color": YELLOW}
        ]

        memory_regions = VGroup()
        labels = VGroup()

        # Create regions
        total_memory = 1024

        for region in regions:
            width = (region["length"] / total_memory) * memory_bg.width
            x_offset = -memory_bg.width/2 + (region["start"] / total_memory) * memory_bg.width + width/2

            rect = Rectangle(width=width, height=1.5, fill_opacity=0.7, color=region["color"])
            rect.move_to([x_offset, 0, 0])

            memory_regions.add(rect)

            # Create label
            label = Text(region["name"], font_size=20)
            label.next_to(rect, DOWN, buff=0.3)

            addr_label = Text(f"({region['start']}-{region['start']+region['length']-1})", font_size=16)
            addr_label.next_to(label, DOWN, buff=0.1)

            labels.add(VGroup(label, addr_label))

        memory_group = VGroup(memory_bg, memory_regions, labels)
        memory_group.move_to(ORIGIN)

        # Animate memory visualization
        self.play(Create(memory_bg))
        self.play(
            *[Create(region) for region in memory_regions],
            run_time=2
        )
        self.play(
            *[Write(label_group) for label_group in labels],
            run_time=2
        )

        # Explain memory regions
        explanation = VGroup(
            Text("• Basic Operations (0-127): Arithmetic and basic instructions", font_size=24),
            Text("• Loops & Branching (256-511): Control flow and decision making", font_size=24),
            Text("• Memory Access (512-767): Data storage and retrieval operations", font_size=24),
            Text("• Advanced Functions (768-1023): Complex algorithms and procedures", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        explanation.scale(0.8)
        explanation.next_to(memory_group, DOWN, buff=0.8)

        for i, line in enumerate(explanation):
            self.play(
                Write(line),
                memory_regions[i].animate.set_fill(opacity=0.9),
                run_time=1
            )
            self.play(memory_regions[i].animate.set_fill(opacity=0.7))

        self.wait(2)

        # Transition to battle
        self.play(
            FadeOut(title),
            FadeOut(memory_group),
            FadeOut(explanation)
        )

    def battle_rounds(self, battle_data, battle_log):
        # Parse rounds from battle log
        rounds = []
        current_round = {}
        attack_code = []
        in_attack_code = False
        execution_results = {}

        # Remove ANSI color codes from battle log
        battle_log = re.sub(r'\x1b\[\d+m', '', battle_log)

        for line in battle_log.split('\n'):
            line = line.strip()

            # New round detection - more lenient matching
            if "ROUND" in line and "==" in line:
                round_match = re.search(r'ROUND\s+(\d+)', line)
                if round_match:
                    if current_round and attack_code:
                        current_round["attack_code"] = "\n".join(attack_code)
                        current_round.update(execution_results)
                        rounds.append(current_round)

                    current_round = {"round": int(round_match.group(1))}
                    attack_code = []
                    in_attack_code = False
                    execution_results = {}

            # Attacker/Defender detection - more lenient matching
            elif "attacks" in line and not in_attack_code:
                attacker_match = re.search(r'(\w+)\s+attacks\s+(\w+)', line)
                if attacker_match:
                    current_round["attacker"] = attacker_match.group(1)
                    current_round["defender"] = attacker_match.group(2)

            # Attack code section
            elif "Attack Code:" in line:
                in_attack_code = True
                continue  # Skip the header line
            elif "Execution Result:" in line:
                in_attack_code = False
            elif in_attack_code:
                attack_code.append(line)

            # Result information
            elif "Instructions executed:" in line:
                val = re.search(r'Instructions executed: (\d+)', line)
                if val:
                    execution_results["instructions"] = int(val.group(1))
            elif "Memory locations corrupted:" in line:
                val = re.search(r'Memory locations corrupted: (\d+)', line)
                if val:
                    execution_results["corrupted"] = int(val.group(1))
            elif "Error:" in line:
                val = re.search(r'Error: (.+)', line)
                if val:
                    execution_results["error"] = val.group(1)
            elif "Attack cost:" in line:
                val = re.search(r'Attack cost: (\d+)', line)
                if val:
                    execution_results["cost"] = int(val.group(1))
            elif "Remaining budget:" in line:
                val = re.search(r'Remaining budget: (\d+)', line)
                if val:
                    execution_results["remaining_budget"] = int(val.group(1))

        # Add the last round
        if current_round and attack_code:
            current_round["attack_code"] = "\n".join(attack_code)
            current_round.update(execution_results)
            rounds.append(current_round)

        # Get all valid rounds
        valid_rounds = [r for r in rounds if all(key in r for key in ["attacker", "defender", "attack_code"])]

        # Use all valid rounds (or at least the first 10)
        interesting_rounds = valid_rounds[:min(10, len(valid_rounds))]

        # Setup memory regions
        memory_regions = {
            "basic_ops": MemoryRegion("Basic Operations", 0, 128, color=BLUE_B),
            "loops": MemoryRegion("Loops/Branching", 256, 256, color=GREEN_B),
            "memory_ops": MemoryRegion("Memory Access", 512, 256, color=RED_B),
            "advanced_ops": MemoryRegion("Advanced Functions", 768, 256, color=YELLOW_B)
        }

        # Position the memory regions
        memory_group = VGroup()

        for name, region in memory_regions.items():
            full_region = region.get_full_region()
            if not memory_group:
                full_region.move_to(LEFT * 3)
            else:
                full_region.next_to(memory_group, DOWN, buff=0.3)
            memory_group.add(full_region)

        memory_group.scale(0.7)
        memory_group.to_edge(LEFT, buff=3)

        # Create title
        battle_title = Text("Battle Rounds", font_size=36)
        battle_title.to_edge(UP, buff=1.5)

        self.play(Write(battle_title))

        # Create and display the memory layout
        self.play(
            *[Create(region) for region in memory_regions.values()],
            *[Write(region.label) for region in memory_regions.values()],
            *[Create(region.health_bar_bg) for region in memory_regions.values()],
            *[Create(region.health_bar) for region in memory_regions.values()],
            run_time=2
        )

        # Optionally show memory cells with a fade in
        self.play(
            *[FadeIn(region.cells) for region in memory_regions.values()],
            run_time=1
        )

        # Visualize each selected round
        for round_data in interesting_rounds:
            self.visualize_battle_round(round_data, memory_regions, memory_group)

        # Cleanup for final results
        self.play(
            FadeOut(battle_title),
            FadeOut(memory_group)
        )
    def visualize_battle_round(self, round_data, memory_regions, memory_group):
        round_num = round_data["round"]
        attacker = round_data["attacker"]
        defender = round_data["defender"]
        attack_code = round_data["attack_code"]
        corrupted = round_data.get("corrupted", 0)
        cost = round_data.get("cost", 0)
        error = round_data.get("error", "None")

        # Clear any previous memory group modifications
        # This prevents the accumulation of green platforms
        for mob in self.mobjects[:]:
            if isinstance(mob, VGroup) and mob != memory_group:
                self.remove(mob)
        

        # Colors based on attacker
        attacker_color = BLUE if attacker == "Claude" else GREEN

        # Position memory regions at LEFT
        memory_group.move_to(LEFT * 8)

        # Show round header pushed higher up
        round_header = Text(f"Round {round_num}: {attacker} attacks {defender}!",
                           font_size=32, color=attacker_color)
        round_header.to_edge(UP, buff=0.5)  # Smaller buffer to push it higher

        # Battle Rounds text positioned at top center
        battle_title = Text("Battle Rounds", font_size=36)
        battle_title.to_edge(UP, buff=0.1)  # Very small buffer to push it to the very top

        self.play(Write(battle_title))
        self.play(Write(round_header))

        # Position Attack Code box at RIGHT
        code_box = Rectangle(width=6, height=5, fill_opacity=0.2, color=GREY)
        code_box.move_to(RIGHT * 8)

        code_title = Text("Attack Code", font_size=28)
        code_title.next_to(code_box, UP, buff=0.2)

        code = AssemblyCode(attack_code)
        code.move_to(code_box.get_center())

        self.play(
            Create(code_box),
            Write(code_title),
            Write(code)
        )

        # Simple straight arrow connecting memory to code
        attack_arrow = Arrow(
            start=memory_group.get_right(),
            end=code_box.get_left(),
            buff=0.5,
            color=attacker_color
        )

        attack_label = Text("Attack", font_size=24, color=attacker_color)
        attack_label.next_to(attack_arrow, UP, buff=0.1)

        self.play(
            Write(attack_label),
            GrowArrow(attack_arrow)
        )

        # Create attack animation particles
        num_particles = max(10, min(corrupted // 5, 40))
        particles = VGroup()

        for _ in range(num_particles):
            dot = Dot(color=attacker_color, radius=0.05)
            dot.move_to(memory_group.get_right())
            particles.add(dot)

        # Create animations for each particle
        particle_animations = []
        corrupted_cells = []

        # Get list of all memory regions
        target_memory = list(memory_regions.values())

        for i, particle in enumerate(particles):
            # Pick a random region and a random cell in that region
            target_region = random.choice(target_memory)
            target_cell_idx = random.randint(0, target_region.length - 1)
            target_cell = target_region.cells[target_cell_idx]

            anim = MoveAlongPath(
                particle,
                CubicBezier(
                    memory_group.get_right(),
                    memory_group.get_right() + UP*random.uniform(-1, 1) + RIGHT*random.uniform(0, 1),
                    code_box.get_left() + UP*random.uniform(-1, 1) + LEFT*random.uniform(0, 1),
                    code_box.get_left()
                ),
                rate_func=rate_functions.ease_in_out_quad,
                run_time=random.uniform(0.5, 1.5)
            )
            particle_animations.append(anim)
            corrupted_cells.append((target_region, target_cell_idx))

        # Play particle animations
        if particle_animations:
            self.play(
                *particle_animations,
                run_time=2
            )

            # Corrupt memory cells and update health bars
            health_bar_animations = []
            corrupt_animations = []

            for region, cell_idx in corrupted_cells:
                cell = region.cells[cell_idx]
                corrupt_animations.append(cell.animate.set_fill(RED, opacity=0.8))

                # Update health based on corruption level
                region_corrupted = sum(1 for r, _ in corrupted_cells if r == region)
                damage = min(100, region_corrupted / region.length * 100)
                new_health_bar = region.update_health(damage/len(corrupted_cells))
                health_bar_animations.append(Transform(region.health_bar, new_health_bar))

            self.play(
                *corrupt_animations,
                *health_bar_animations,
                run_time=1
            )
        else:
            # Fallback if no animations were created
            self.wait(2)

        # Results box below code box
        results_box = Rectangle(width=6, height=2, fill_opacity=0.2, color=GREY)
        results_box.next_to(code_box, DOWN, buff=0.5)

        results_title = Text("Attack Results", font_size=28)
        results_title.next_to(results_box, UP, buff=0.2)

        results_text = VGroup(
            Text(f"Memory corrupted: {corrupted} locations", font_size=22),
            Text(f"Attack cost: {cost} points", font_size=22),
            Text(f"Error: {error[:30]}{'...' if len(error) > 30 else ''}", font_size=22)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        results_text.move_to(results_box.get_center())

        self.play(
            Create(results_box),
            Write(results_title),
            Write(results_text)
        )

        # Wait and clean up for next round
        self.wait(1)

        # Remove all elements except memory_group
        objects_to_remove = [battle_title, round_header, code_box, code_title, code, 
                            attack_arrow, attack_label, results_box, results_title, 
                            results_text, particles]

        self.play(
            *[FadeOut(obj) for obj in objects_to_remove if obj in self.mobjects]
        )

        # Reset corruption effects in memory cells
        for region in memory_regions.values():
            for cell in region.cells:
                if cell.get_fill_opacity() > 0.3:  # If it was corrupted
                    cell.set_fill(WHITE, opacity=0.3)  # Reset to original state
    
    def final_results(self, battle_data):
        # Create title
        final_title = Text("Battle Results", font_size=48, color=WHITE)
        final_title.to_edge(UP, buff=1.2)
    
        # Get winner and scores
        winner = battle_data.get("winner", "Unknown")
        claude_score = battle_data.get("claude_score", 0)
        gpt_score = battle_data.get("gpt_score", 0)
        claude_budget = battle_data.get("claude_final_budget", 0)
        gpt_budget = battle_data.get("gpt_final_budget", 0)
    
        # Create final text
        winner_text = Text(f"Winner: {winner}!", font_size=42, 
                          color=BLUE if winner == "Claude" else GREEN)
    
        # Create score group with more spacing
        score_group = VGroup(
            Text(f"Claude Score: {claude_score}", font_size=36, color=BLUE),
            Text(f"GPT Score: {gpt_score}", font_size=36, color=GREEN),
            Text(f"Claude Final Budget: {claude_budget}", font_size=30, color=BLUE_B),
            Text(f"GPT Final Budget: {gpt_budget}", font_size=30, color=GREEN_B)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
    
        # Position with more vertical spacing
        winner_text.next_to(final_title, DOWN, buff=1.5)
        score_group.next_to(winner_text, DOWN, buff=1.0)
        
        # Center these elements
        winner_text.move_to(ORIGIN + UP * 3)
        score_group.next_to(winner_text, DOWN, buff=0.8)
    
        # Create memory impact visualization with better spacing
        impact_title = Text("Memory Region Impact", font_size=36)
        impact_title.next_to(score_group, DOWN, buff=2.0)
    
        # Create bar charts for memory impact
        claude_impact = battle_data.get("claude_memory_impact", {})
        gpt_impact = battle_data.get("gpt_memory_impact", {})
    
        # Create axes for Claude's impact with wider bars
        claude_chart = BarChart(
            values=[
                claude_impact.get("basic_ops", 0),
                claude_impact.get("loops", 0),
                claude_impact.get("memory_ops", 0),
                claude_impact.get("advanced_ops", 0)
            ],
            bar_names=["Basic Ops", "Loops", "Memory Ops", "Advanced"],
            y_range=[0, 1500, 300],
            x_length=14,  # Wider chart
            y_length=5,
            bar_colors=[BLUE, GREEN, RED, YELLOW]
        )
    
        # Create axes for GPT's impact with wider bars
        gpt_chart = BarChart(
            values=[
                gpt_impact.get("basic_ops", 0),
                gpt_impact.get("loops", 0),
                gpt_impact.get("memory_ops", 0),
                gpt_impact.get("advanced_ops", 0)
            ],
            bar_names=["Basic Ops", "Loops", "Memory Ops", "Advanced"],
            y_range=[0, 1500, 300],
            x_length=14,  # Wider chart
            y_length=5,
            bar_colors=[BLUE, GREEN, RED, YELLOW]
        )
    
        # Position charts with better spacing
        claude_label = Text("Claude's Impact on GPT Memory", font_size=30, color=BLUE)
        claude_label.next_to(impact_title, DOWN, buff=0.8)
        claude_chart.next_to(claude_label, DOWN, buff=0.8)
    
        gpt_label = Text("GPT's Impact on Claude Memory", font_size=30, color=GREEN)
        gpt_label.next_to(claude_chart, DOWN, buff=1.5)
        gpt_chart.next_to(gpt_label, DOWN, buff=0.8)
    
        # Make the charts wider by spreading them horizontally
        claude_chart.stretch_to_fit_width(18)  # Stretch to use more horizontal space
        gpt_chart.stretch_to_fit_width(18)  # Stretch to use more horizontal space
    
        # Animate final results
        self.play(Write(final_title))
        self.play(Write(winner_text))
        self.play(Write(score_group))
    
        self.wait(1)
    
        self.play(
            Write(impact_title),
            Write(claude_label)
        )
        self.play(Create(claude_chart))
    
        self.wait(1)
    
        self.play(
            Write(gpt_label)
        )
        self.play(Create(gpt_chart))
    
        # Final wait
        self.wait(3)

# Use this code to render the visualization
if __name__ == "__main__":
    # Command line: python -m manim -pql ai_battle_visualization.py AIBattleSimulation
    scene = AIBattleSimulation()
    scene.render()
