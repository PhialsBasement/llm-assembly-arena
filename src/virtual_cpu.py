import logging

logger = logging.getLogger('LLM-ASM-Battle')

class VirtualCPU:
    """Simulated CPU that can execute assembly instructions"""

    # Instruction set - maps assembly mnemonics to their implementations
    INSTRUCTION_SET = {
        "MOV": lambda cpu, args: cpu._mov(*args),
        "ADD": lambda cpu, args: cpu._add(*args),
        "SUB": lambda cpu, args: cpu._sub(*args),
        "MUL": lambda cpu, args: cpu._mul(*args),
        "DIV": lambda cpu, args: cpu._div(*args),
        "AND": lambda cpu, args: cpu._and(*args),
        "OR": lambda cpu, args: cpu._or(*args),
        "XOR": lambda cpu, args: cpu._xor(*args),
        "NOT": lambda cpu, args: cpu._not(*args),
        "NEG": lambda cpu, args: cpu._neg(*args),
        "INC": lambda cpu, args: cpu._inc(*args),
        "DEC": lambda cpu, args: cpu._dec(*args),
        "SHL": lambda cpu, args: cpu._shl(*args),
        "SHR": lambda cpu, args: cpu._shr(*args),
        "ROL": lambda cpu, args: cpu._rol(*args),
        "ROR": lambda cpu, args: cpu._ror(*args),
        "CMP": lambda cpu, args: cpu._cmp(*args),
        "TEST": lambda cpu, args: cpu._test(*args),
        "JMP": lambda cpu, args: cpu._jmp(*args),
        "JE": lambda cpu, args: cpu._je(*args),
        "JNE": lambda cpu, args: cpu._jne(*args),
        "JZ": lambda cpu, args: cpu._je(*args),  # Alias for JE
        "JNZ": lambda cpu, args: cpu._jne(*args),  # Alias for JNE
        "JG": lambda cpu, args: cpu._jg(*args),
        "JGE": lambda cpu, args: cpu._jge(*args),
        "JL": lambda cpu, args: cpu._jl(*args),
        "JLE": lambda cpu, args: cpu._jle(*args),
        "JA": lambda cpu, args: cpu._ja(*args),
        "JAE": lambda cpu, args: cpu._jae(*args),
        "JB": lambda cpu, args: cpu._jb(*args),
        "JBE": lambda cpu, args: cpu._jbe(*args),
        "LOOP": lambda cpu, args: cpu._loop(*args),
        "PUSH": lambda cpu, args: cpu._push(*args),
        "POP": lambda cpu, args: cpu._pop(*args),
        "CALL": lambda cpu, args: cpu._call(*args),
        "RET": lambda cpu, args: cpu._ret(*args),
        "INT": lambda cpu, args: cpu._interrupt(*args),
        "NOP": lambda cpu, args: None,
    }

    def __init__(self, memory_size=1024, owner=None):
        """Initialize CPU with memory and registers"""
        self.memory = [0] * memory_size
        self.memory_size = memory_size
        self.registers = {
            'eax': 0, 'ebx': 0, 'ecx': 0, 'edx': 0,
            'esi': 0, 'edi': 0, 'ebp': 0, 'esp': memory_size - 1,
            'eip': 0, 'eflags': 0
        }
        self.owner = owner  # Which LLM owns this CPU
        self.instructions_executed = 0
        self.max_instructions = 1000  # Prevent infinite loops
        self.error = None
        self.stack_base = memory_size - 128  # Reserve 128 bytes for stack

        # Label table for resolving jump targets
        self.labels = {}

        # Memory protection - mark regions as readable/writable/executable
        self.memory_protection = ['rwx'] * memory_size

        # Memory corruption tracker
        self.corrupted_addresses = set()

        # Flag bits in eflags register
        self.FLAG_ZERO = 0x0001
        self.FLAG_SIGN = 0x0002
        self.FLAG_OVERFLOW = 0x0004
        self.FLAG_CARRY = 0x0008

        # Fill memory with initial pattern for the owner
        self._initialize_memory()

    def _initialize_memory(self):
        """Initialize memory with a pattern specific to the owner"""
        if self.owner == "claude":
            pattern = [0x41, 0x4E, 0x54, 0x48, 0x52, 0x4F, 0x50, 0x49, 0x43]  # "ANTHROPIC" in ASCII
        else:
            pattern = [0x4F, 0x50, 0x45, 0x4E, 0x41, 0x49]  # "OPENAI" in ASCII

        # Fill memory with repeating pattern
        for i in range(0, self.memory_size, len(pattern)):
            for j, val in enumerate(pattern):
                if i + j < self.memory_size:
                    self.memory[i + j] = val

    def load_program(self, program, start_address=0):
        """Load program (list of bytes) into memory at start_address"""
        for i, byte in enumerate(program):
            if start_address + i < self.memory_size:
                self.memory[start_address + i] = byte

    def _preprocess_program(self, start_address=0):
        """First pass to build label table by scanning through the program"""
        current_addr = start_address

        # Reset labels table
        self.labels = {}

        # First pass: collect all labels
        lines = []
        line_addresses = []

        # Read the entire program into lines
        while current_addr < self.memory_size:
            line = ""
            line_start = current_addr

            # Read a line
            while current_addr < self.memory_size and self.memory[current_addr] not in (0, 0x0A):
                line += chr(self.memory[current_addr])
                current_addr += 1

            # Skip null or newline
            if current_addr < self.memory_size and self.memory[current_addr] in (0, 0x0A):
                current_addr += 1

            line = line.strip()
            if line:  # Only consider non-empty lines
                lines.append(line)
                line_addresses.append(line_start)

        # Second pass: find labels and their target addresses
        for i, line in enumerate(lines):
            # Check if line contains a label (ends with colon)
            if ':' in line:
                parts = line.split(':', 1)
                label_name = parts[0].strip().lower()

                # Label points to this line or the next instruction
                target_addr = line_addresses[i]

                # If the label is followed by an instruction on the same line, 
                # the target is this line, otherwise it's the next line
                if len(parts) > 1 and parts[1].strip():
                    # Label and instruction on same line
                    target_addr = line_addresses[i]
                elif i < len(lines) - 1:
                    # Label alone, point to next line
                    target_addr = line_addresses[i + 1]

                self.labels[label_name] = target_addr
                logger.info(f"Found label: {label_name} at address 0x{target_addr:X}")

        # Log all found labels
        logger.info(f"Label table: {self.labels}")
        return len(self.labels) > 0

    def execute(self, start_address=0, max_instructions=None):
        """Execute program starting at address"""
        # First pass: build label table
        self._preprocess_program(start_address)

        self.registers['eip'] = start_address
        self.instructions_executed = 0
        self.error = None

        if max_instructions is not None:
            self.max_instructions = max_instructions

        while self.registers['eip'] < self.memory_size and self.instructions_executed < self.max_instructions:
            # Get instruction at current eip
            try:
                instruction = self._fetch_instruction()
                if instruction is None:
                    break

                # Parse instruction
                opcode, args = self._decode_instruction(instruction)

                # Log instruction for debugging
                logger.debug(f"Execute: {opcode} {', '.join(args)}")

                # Execute instruction
                if opcode in self.INSTRUCTION_SET:
                    self.INSTRUCTION_SET[opcode](self, args)
                else:
                    self.error = f"Unknown opcode: {opcode}"
                    break

                self.instructions_executed += 1

                # Check if instruction wrote to protected memory
                if self.error:
                    break

            except Exception as e:
                self.error = f"Execution error: {str(e)}"
                logger.error(f"CPU exception: {str(e)}")
                break

        if self.instructions_executed >= self.max_instructions and not self.error:
            self.error = "Maximum instruction limit reached (possible infinite loop)"

        return {
            'instructions_executed': self.instructions_executed,
            'error': self.error,
            'corrupted_addresses': len(self.corrupted_addresses)
        }

    def _fetch_instruction(self):
        """Fetch instruction at current eip"""
        instruction = ""
        eip = self.registers['eip']

        # Skip to next line if we're at the end of a line
        while eip < self.memory_size and self.memory[eip] == 0x0A:  # newline
            eip += 1
            self.registers['eip'] = eip

        # Read until newline or null terminator
        while eip < self.memory_size and self.memory[eip] not in (0, 0x0A):
            instruction += chr(self.memory[eip])
            eip += 1

        # Update eip to next instruction
        if eip < self.memory_size and self.memory[eip] == 0x0A:
            eip += 1

        self.registers['eip'] = eip

        return instruction.strip() if instruction else None

    def _decode_instruction(self, instruction_str):
        """Parse an assembly instruction into opcode and arguments"""
        # Handle comments
        if ';' in instruction_str:
            instruction_str = instruction_str.split(';', 1)[0].strip()

        if not instruction_str:
            return "NOP", []

        # Skip labels (lines ending with :)
        if instruction_str.endswith(':'):
            return "NOP", []

        parts = instruction_str.split(None, 1)
        opcode = parts[0].upper()

        args = []
        if len(parts) > 1 and parts[1].strip():
            # Split args by commas, but respect parentheses
            arg_parts = []
            current_arg = ""
            paren_level = 0

            for char in parts[1]:
                if char == ',' and paren_level == 0:
                    arg_parts.append(current_arg.strip())
                    current_arg = ""
                else:
                    if char == '(':
                        paren_level += 1
                    elif char == ')':
                        paren_level -= 1
                    current_arg += char

            if current_arg.strip():
                arg_parts.append(current_arg.strip())

            args = [arg.strip() for arg in arg_parts]

        return opcode, args

    def _get_value(self, arg):
        """Get value from register, memory, or immediate"""
        # Handle sub-registers like AL, AH, etc.
        if arg.lower() == 'al':
            return self.registers['eax'] & 0xFF
        elif arg.lower() == 'ah':
            return (self.registers['eax'] >> 8) & 0xFF
        elif arg.lower() == 'ax':
            return self.registers['eax'] & 0xFFFF
        elif arg.lower() == 'bl':
            return self.registers['ebx'] & 0xFF
        elif arg.lower() == 'bh':
            return (self.registers['ebx'] >> 8) & 0xFF
        elif arg.lower() == 'bx':
            return self.registers['ebx'] & 0xFFFF
        elif arg.lower() == 'cl':
            return self.registers['ecx'] & 0xFF
        elif arg.lower() == 'ch':
            return (self.registers['ecx'] >> 8) & 0xFF
        elif arg.lower() == 'cx':
            return self.registers['ecx'] & 0xFFFF
        elif arg.lower() == 'dl':
            return self.registers['edx'] & 0xFF
        elif arg.lower() == 'dh':
            return (self.registers['edx'] >> 8) & 0xFF
        elif arg.lower() == 'dx':
            return self.registers['edx'] & 0xFFFF
        # Full register
        elif arg.lower() in self.registers:
            return self.registers[arg.lower()]

        # Label reference
        elif arg.lower() in self.labels:
            return self.labels[arg.lower()]

        # Memory address in brackets [addr]
        if arg.startswith('[') and arg.endswith(']'):
            addr_expr = arg[1:-1].strip()

            # Base register addressing [reg]
            if addr_expr.lower() in self.registers:
                addr = self.registers[addr_expr.lower()]
                if 0 <= addr < self.memory_size:
                    return self.memory[addr]
                else:
                    self.error = f"Memory access out of bounds: {addr}"
                    return 0

            # Base + offset addressing [reg+offset]
            if '+' in addr_expr:
                parts = addr_expr.split('+')
                if len(parts) >= 2:
                    base = parts[0].strip()
                    offset = parts[1].strip()

                    base_val = 0
                    if base.lower() in self.registers:
                        base_val = self.registers[base.lower()]
                    elif base.lower() in self.labels:
                        base_val = self.labels[base.lower()]

                    offset_val = self._parse_immediate(offset)

                    addr = base_val + offset_val
                    if 0 <= addr < self.memory_size:
                        return self.memory[addr]
                    else:
                        self.error = f"Memory access out of bounds: {addr}"
                        return 0

            # Base - offset addressing [reg-offset]
            if '-' in addr_expr:
                parts = addr_expr.split('-')
                if len(parts) >= 2:
                    base = parts[0].strip()
                    offset = parts[1].strip()

                    base_val = 0
                    if base.lower() in self.registers:
                        base_val = self.registers[base.lower()]
                    elif base.lower() in self.labels:
                        base_val = self.labels[base.lower()]

                    offset_val = self._parse_immediate(offset)

                    addr = base_val - offset_val
                    if 0 <= addr < self.memory_size:
                        return self.memory[addr]
                    else:
                        self.error = f"Memory access out of bounds: {addr}"
                        return 0

            # Scaled index addressing [base+index*scale+offset]
            if '*' in addr_expr:
                # This is a complex addressing mode like [eax+ecx*4+20]
                # Parse each component
                base_val = 0
                index_val = 0
                scale_val = 1
                offset_val = 0

                # Parse the expression - this is simplified and not fully robust
                parts = addr_expr.replace('+', ' + ').replace('*', ' * ').replace('-', ' - ').split()

                i = 0
                while i < len(parts):
                    token = parts[i].strip()

                    if token in self.registers:
                        if i + 2 < len(parts) and parts[i+1] == '*':
                            # This is an index*scale component
                            index_val = self.registers[token]
                            scale_val = self._parse_immediate(parts[i+2])
                            i += 3
                        else:
                            # This is a base register
                            base_val = self.registers[token]
                            i += 1
                    elif token in ('+', '-'):
                        sign = 1 if token == '+' else -1
                        if i + 1 < len(parts):
                            next_token = parts[i+1]
                            if next_token in self.registers:
                                # +/- register
                                base_val += sign * self.registers[next_token]
                            else:
                                # +/- immediate
                                offset_val += sign * self._parse_immediate(next_token)
                            i += 2
                        else:
                            self.error = f"Invalid addressing mode: {addr_expr}"
                            return 0
                    else:
                        # Direct offset
                        offset_val = self._parse_immediate(token)
                        i += 1

                addr = base_val + (index_val * scale_val) + offset_val
                if 0 <= addr < self.memory_size:
                    return self.memory[addr]
                else:
                    self.error = f"Memory access out of bounds: {addr}"
                    return 0

            # Direct addressing [addr]
            try:
                addr = self._parse_immediate(addr_expr)
                if 0 <= addr < self.memory_size:
                    return self.memory[addr]
                else:
                    self.error = f"Memory access out of bounds: {addr}"
                    return 0
            except ValueError:
                # Check if it's a label
                if addr_expr.lower() in self.labels:
                    addr = self.labels[addr_expr.lower()]
                    if 0 <= addr < self.memory_size:
                        return self.memory[addr]
                    else:
                        self.error = f"Memory access out of bounds: {addr}"
                        return 0
                else:
                    self.error = f"Invalid memory address: {addr_expr}"
                    return 0

        # Immediate value
        return self._parse_immediate(arg)
    
    def _parse_immediate(self, value_str):
        """Parse an immediate value from string"""
        # Check if it's a label first
        if value_str.lower() in self.labels:
            return self.labels[value_str.lower()]
    
        # Handle character literals
        if value_str.startswith("'") and value_str.endswith("'") and len(value_str) == 3:
            return ord(value_str[1])  # Return ASCII value of the character
    
        # Try to parse as a number
        try:
            if value_str.startswith('0x'):
                return int(value_str, 16)
            elif value_str.startswith('0b'):
                return int(value_str, 2)
            elif value_str.startswith('0o'):
                return int(value_str, 8)
            else:
                return int(value_str)
        except ValueError:
            self.error = f"Invalid immediate value: {value_str}"
            return 0

    def _set_value(self, dest, value):
        """Set value to register or memory"""
        # Handle sub-registers like AL, AH, etc.
        if dest.lower() == 'al':
            # Preserve other bits, only modify lowest 8 bits
            self.registers['eax'] = (self.registers['eax'] & 0xFFFFFF00) | (value & 0xFF)
            return True
        elif dest.lower() == 'ah':
            # Preserve other bits, only modify second lowest 8 bits
            self.registers['eax'] = (self.registers['eax'] & 0xFFFF00FF) | ((value & 0xFF) << 8)
            return True
        elif dest.lower() == 'ax':
            # Preserve other bits, only modify lowest 16 bits
            self.registers['eax'] = (self.registers['eax'] & 0xFFFF0000) | (value & 0xFFFF)
            return True
        elif dest.lower() == 'bl':
            self.registers['ebx'] = (self.registers['ebx'] & 0xFFFFFF00) | (value & 0xFF)
            return True
        elif dest.lower() == 'bh':
            self.registers['ebx'] = (self.registers['ebx'] & 0xFFFF00FF) | ((value & 0xFF) << 8)
            return True
        elif dest.lower() == 'bx':
            self.registers['ebx'] = (self.registers['ebx'] & 0xFFFF0000) | (value & 0xFFFF)
            return True
        elif dest.lower() == 'cl':
            self.registers['ecx'] = (self.registers['ecx'] & 0xFFFFFF00) | (value & 0xFF)
            return True
        elif dest.lower() == 'ch':
            self.registers['ecx'] = (self.registers['ecx'] & 0xFFFF00FF) | ((value & 0xFF) << 8)
            return True
        elif dest.lower() == 'cx':
            self.registers['ecx'] = (self.registers['ecx'] & 0xFFFF0000) | (value & 0xFFFF)
            return True
        elif dest.lower() == 'dl':
            self.registers['edx'] = (self.registers['edx'] & 0xFFFFFF00) | (value & 0xFF)
            return True
        elif dest.lower() == 'dh':
            self.registers['edx'] = (self.registers['edx'] & 0xFFFF00FF) | ((value & 0xFF) << 8)
            return True
        elif dest.lower() == 'dx':
            self.registers['edx'] = (self.registers['edx'] & 0xFFFF0000) | (value & 0xFFFF)
            return True
        # Full register
        elif dest.lower() in self.registers:
            self.registers[dest.lower()] = value & 0xFFFFFFFF  # Apply 32-bit mask
            return True

        # Memory address in brackets [addr]
        if dest.startswith('[') and dest.endswith(']'):
            addr_expr = dest[1:-1].strip()

            # Base register addressing [reg]
            if addr_expr.lower() in self.registers:
                addr = self.registers[addr_expr.lower()]
                if 0 <= addr < self.memory_size:
                    prev_value = self.memory[addr]
                    self.memory[addr] = value & 0xFF  # Apply 8-bit mask

                    # Track corruption if memory was initialized and now changed
                    if prev_value != self.memory[addr]:
                        self.corrupted_addresses.add(addr)

                    return True
                else:
                    self.error = f"Memory write out of bounds: {addr}"
                    return False

            # Base + offset addressing [reg+offset]
            if '+' in addr_expr:
                parts = addr_expr.split('+')
                if len(parts) >= 2:
                    base = parts[0].strip()
                    offset = parts[1].strip()

                    base_val = 0
                    if base.lower() in self.registers:
                        base_val = self.registers[base.lower()]
                    elif base.lower() in self.labels:
                        base_val = self.labels[base.lower()]

                    offset_val = self._parse_immediate(offset)

                    addr = base_val + offset_val
                    if 0 <= addr < self.memory_size:
                        prev_value = self.memory[addr]
                        self.memory[addr] = value & 0xFF

                        # Track corruption if memory was initialized and now changed
                        if prev_value != self.memory[addr]:
                            self.corrupted_addresses.add(addr)

                        return True
                    else:
                        self.error = f"Memory write out of bounds: {addr}"
                        return False

            # Base - offset addressing [reg-offset]
            if '-' in addr_expr:
                parts = addr_expr.split('-')
                if len(parts) >= 2:
                    base = parts[0].strip()
                    offset = parts[1].strip()

                    base_val = 0
                    if base.lower() in self.registers:
                        base_val = self.registers[base.lower()]
                    elif base.lower() in self.labels:
                        base_val = self.labels[base.lower()]

                    offset_val = self._parse_immediate(offset)

                    addr = base_val - offset_val
                    if 0 <= addr < self.memory_size:
                        prev_value = self.memory[addr]
                        self.memory[addr] = value & 0xFF

                        # Track corruption if memory was initialized and now changed
                        if prev_value != self.memory[addr]:
                            self.corrupted_addresses.add(addr)

                        return True
                    else:
                        self.error = f"Memory write out of bounds: {addr}"
                        return False

            # Scaled index addressing [base+index*scale+offset]
            if '*' in addr_expr:
                # This is a complex addressing mode like [eax+ecx*4+20]
                # Parse each component
                base_val = 0
                index_val = 0
                scale_val = 1
                offset_val = 0

                # Parse the expression - this is simplified and not fully robust
                parts = addr_expr.replace('+', ' + ').replace('*', ' * ').replace('-', ' - ').split()

                i = 0
                while i < len(parts):
                    token = parts[i].strip()

                    if token in self.registers:
                        if i + 2 < len(parts) and parts[i+1] == '*':
                            # This is an index*scale component
                            index_val = self.registers[token]
                            scale_val = self._parse_immediate(parts[i+2])
                            i += 3
                        else:
                            # This is a base register
                            base_val = self.registers[token]
                            i += 1
                    elif token in ('+', '-'):
                        sign = 1 if token == '+' else -1
                        if i + 1 < len(parts):
                            next_token = parts[i+1]
                            if next_token in self.registers:
                                # +/- register
                                base_val += sign * self.registers[next_token]
                            else:
                                # +/- immediate
                                offset_val += sign * self._parse_immediate(next_token)
                            i += 2
                        else:
                            self.error = f"Invalid addressing mode: {addr_expr}"
                            return False
                    else:
                        # Direct offset
                        offset_val = self._parse_immediate(token)
                        i += 1

                addr = base_val + (index_val * scale_val) + offset_val
                if 0 <= addr < self.memory_size:
                    prev_value = self.memory[addr]
                    self.memory[addr] = value & 0xFF

                    # Track corruption if memory was initialized and now changed
                    if prev_value != self.memory[addr]:
                        self.corrupted_addresses.add(addr)

                    return True
                else:
                    self.error = f"Memory write out of bounds: {addr}"
                    return False

            # Direct addressing [addr]
            try:
                addr = self._parse_immediate(addr_expr)
                if 0 <= addr < self.memory_size:
                    prev_value = self.memory[addr]
                    self.memory[addr] = value & 0xFF

                    # Track corruption if memory was initialized and now changed
                    if prev_value != self.memory[addr]:
                        self.corrupted_addresses.add(addr)

                    return True
                else:
                    self.error = f"Memory write out of bounds: {addr}"
                    return False
            except ValueError:
                # Check if it's a label
                if addr_expr.lower() in self.labels:
                    addr = self.labels[addr_expr.lower()]
                    if 0 <= addr < self.memory_size:
                        prev_value = self.memory[addr]
                        self.memory[addr] = value & 0xFF

                        # Track corruption if memory was initialized and now changed
                        if prev_value != self.memory[addr]:
                            self.corrupted_addresses.add(addr)

                        return True
                    else:
                        self.error = f"Memory write out of bounds: {addr}"
                        return False
                else:
                    self.error = f"Invalid memory address: {addr_expr}"
                    return False

        self.error = f"Invalid destination: {dest}"
        return False

    # Instruction implementations
    def _mov(self, dest, src):
        """Move value from src to dest"""
        value = self._get_value(src)
        if self.error:
            return

        self._set_value(dest, value)

    def _add(self, dest, src):
        """Add src to dest"""
        dest_val = self._get_value(dest)
        src_val = self._get_value(src)
        if self.error:
            return

        result = (dest_val + src_val) & 0xFFFFFFFF
        self._set_value(dest, result)

        # Update flags
        self._update_flags(result, dest_val, src_val, 'add')

    def _sub(self, dest, src):
        """Subtract src from dest"""
        dest_val = self._get_value(dest)
        src_val = self._get_value(src)
        if self.error:
            return

        result = (dest_val - src_val) & 0xFFFFFFFF
        self._set_value(dest, result)

        # Update flags
        self._update_flags(result, dest_val, src_val, 'sub')

    def _mul(self, dest, src=None):
        """Multiply EAX by src or dest"""
        if src is None:
            # Single operand form: MUL src
            src_val = self._get_value(dest)  # dest is actually the source
            eax_val = self.registers['eax']

            result = (eax_val * src_val) & 0xFFFFFFFF
            self.registers['eax'] = result & 0xFFFFFFFF
            self.registers['edx'] = (result >> 32) & 0xFFFFFFFF

            # Update flags (simplified)
            self._update_flags(result, eax_val, src_val, 'mul')
        else:
            # Two operand form: MUL dest, src
            dest_val = self._get_value(dest)
            src_val = self._get_value(src)
            if self.error:
                return

            result = (dest_val * src_val) & 0xFFFFFFFF
            self._set_value(dest, result)

            # Update flags
            self._update_flags(result, dest_val, src_val, 'mul')

    def _div(self, dest, src=None):
        """Divide EDX:EAX by src or dest/src"""
        if src is None:
            # Single operand form: DIV src (divides EDX:EAX by src)
            divisor = self._get_value(dest)  # dest is actually the source
            if divisor == 0:
                self.error = "Division by zero"
                return

            dividend = (self.registers['edx'] << 32) | self.registers['eax']
            quotient = dividend // divisor
            remainder = dividend % divisor

            if quotient > 0xFFFFFFFF:
                self.error = "Division overflow"
                return

            self.registers['eax'] = quotient & 0xFFFFFFFF
            self.registers['edx'] = remainder & 0xFFFFFFFF
        else:
            # Two operand form: DIV dest, src
            dest_val = self._get_value(dest)
            src_val = self._get_value(src)
            if self.error:
                return

            if src_val == 0:
                self.error = "Division by zero"
                return

            result = (dest_val // src_val) & 0xFFFFFFFF
            self._set_value(dest, result)

    def _and(self, dest, src):
        """Bitwise AND dest with src"""
        dest_val = self._get_value(dest)
        src_val = self._get_value(src)
        if self.error:
            return

        result = dest_val & src_val
        self._set_value(dest, result)

        # Update flags
        self._update_flags(result, dest_val, src_val, 'logic')

    def _or(self, dest, src):
        """Bitwise OR dest with src"""
        dest_val = self._get_value(dest)
        src_val = self._get_value(src)
        if self.error:
            return

        result = dest_val | src_val
        self._set_value(dest, result)

        # Update flags
        self._update_flags(result, dest_val, src_val, 'logic')

    def _xor(self, dest, src):
        """Bitwise XOR dest with src"""
        dest_val = self._get_value(dest)
        src_val = self._get_value(src)
        if self.error:
            return

        result = dest_val ^ src_val
        self._set_value(dest, result)

        # Update flags
        self._update_flags(result, dest_val, src_val, 'logic')

    def _not(self, dest):
        """Bitwise NOT dest"""
        dest_val = self._get_value(dest)
        if self.error:
            return

        result = ~dest_val & 0xFFFFFFFF
        self._set_value(dest, result)

        # Update flags
        self._update_flags(result, dest_val, 0, 'logic')

    def _neg(self, dest):
        """Two's complement negation of dest"""
        dest_val = self._get_value(dest)
        if self.error:
            return

        result = (-dest_val) & 0xFFFFFFFF
        self._set_value(dest, result)

        # Update flags
        self._update_flags(result, dest_val, 0, 'neg')

    def _inc(self, dest):
        """Increment dest by 1"""
        dest_val = self._get_value(dest)
        if self.error:
            return

        result = (dest_val + 1) & 0xFFFFFFFF
        self._set_value(dest, result)

        # Update flags
        self._update_flags(result, dest_val, 1, 'add')

    def _dec(self, dest):
        """Decrement dest by 1"""
        dest_val = self._get_value(dest)
        if self.error:
            return

        result = (dest_val - 1) & 0xFFFFFFFF
        self._set_value(dest, result)

        # Update flags
        self._update_flags(result, dest_val, 1, 'sub')

    def _shl(self, dest, count):
        """Shift left dest by count bits"""
        dest_val = self._get_value(dest)
        count_val = self._get_value(count)
        if self.error:
            return

        # Limit shift count to 31 to prevent excessive shifts
        count_val = min(count_val, 31)

        result = (dest_val << count_val) & 0xFFFFFFFF
        self._set_value(dest, result)

        # Update flags
        self._update_flags(result, dest_val, count_val, 'shift')

    def _shr(self, dest, count):
        """Shift right dest by count bits"""
        dest_val = self._get_value(dest)
        count_val = self._get_value(count)
        if self.error:
            return

        # Limit shift count to 31 to prevent excessive shifts
        count_val = min(count_val, 31)

        result = (dest_val >> count_val) & 0xFFFFFFFF
        self._set_value(dest, result)

        # Update flags
        self._update_flags(result, dest_val, count_val, 'shift')

    def _rol(self, dest, count):
        """Rotate left dest by count bits"""
        dest_val = self._get_value(dest)
        count_val = self._get_value(count) % 32  # Modulo 32 for rotation
        if self.error:
            return

        # Perform rotation
        result = ((dest_val << count_val) | (dest_val >> (32 - count_val))) & 0xFFFFFFFF
        self._set_value(dest, result)

        # Set carry flag to the bit that was rotated
        if count_val > 0:
            self.registers['eflags'] &= ~self.FLAG_CARRY  # Clear carry flag
            if result & 1:  # Check if LSB is 1
                self.registers['eflags'] |= self.FLAG_CARRY

    def _ror(self, dest, count):
        """Rotate right dest by count bits"""
        dest_val = self._get_value(dest)
        count_val = self._get_value(count) % 32  # Modulo 32 for rotation
        if self.error:
            return

        # Perform rotation
        result = ((dest_val >> count_val) | (dest_val << (32 - count_val))) & 0xFFFFFFFF
        self._set_value(dest, result)

        # Set carry flag to the bit that was rotated
        if count_val > 0:
            self.registers['eflags'] &= ~self.FLAG_CARRY  # Clear carry flag
            if result & 0x80000000:  # Check if MSB is 1
                self.registers['eflags'] |= self.FLAG_CARRY

    def _cmp(self, op1, op2):
        """Compare op1 and op2 and set flags"""
        op1_val = self._get_value(op1)
        op2_val = self._get_value(op2)
        if self.error:
            return

        # Perform subtraction but don't save result
        result = (op1_val - op2_val) & 0xFFFFFFFF

        # Update flags
        self._update_flags(result, op1_val, op2_val, 'sub')

    def _test(self, op1, op2):
        """Bitwise AND of op1 and op2 and set flags"""
        op1_val = self._get_value(op1)
        op2_val = self._get_value(op2)
        if self.error:
            return

        # Perform AND but don't save result
        result = op1_val & op2_val

        # Update flags
        self._update_flags(result, op1_val, op2_val, 'logic')

    def _jmp(self, target):
        """Jump to target address"""
        # Check if target is a label
        if target.lower() in self.labels:
            self.registers['eip'] = self.labels[target.lower()]
            return

        # Otherwise, try to get value from register or immediate
        target_val = self._get_value(target)
        if self.error:
            return

        if 0 <= target_val < self.memory_size:
            self.registers['eip'] = target_val
        else:
            self.error = f"Jump target out of bounds: {target_val}"

    def _je(self, target):
        """Jump if equal (zero flag set)"""
        if self.registers['eflags'] & self.FLAG_ZERO:
            self._jmp(target)

    def _jne(self, target):
        """Jump if not equal (zero flag clear)"""
        if not (self.registers['eflags'] & self.FLAG_ZERO):
            self._jmp(target)

    def _jg(self, target):
        """Jump if greater (zero flag clear and sign == overflow)"""
        zero = bool(self.registers['eflags'] & self.FLAG_ZERO)
        sign = bool(self.registers['eflags'] & self.FLAG_SIGN)
        overflow = bool(self.registers['eflags'] & self.FLAG_OVERFLOW)

        if not zero and (sign == overflow):
            self._jmp(target)

    def _jge(self, target):
        """Jump if greater or equal (sign == overflow)"""
        sign = bool(self.registers['eflags'] & self.FLAG_SIGN)
        overflow = bool(self.registers['eflags'] & self.FLAG_OVERFLOW)

        if sign == overflow:
            self._jmp(target)

    def _jl(self, target):
        """Jump if less (sign != overflow)"""
        sign = bool(self.registers['eflags'] & self.FLAG_SIGN)
        overflow = bool(self.registers['eflags'] & self.FLAG_OVERFLOW)

        if sign != overflow:
            self._jmp(target)

    def _jle(self, target):
        """Jump if less or equal (zero flag set or sign != overflow)"""
        zero = bool(self.registers['eflags'] & self.FLAG_ZERO)
        sign = bool(self.registers['eflags'] & self.FLAG_SIGN)
        overflow = bool(self.registers['eflags'] & self.FLAG_OVERFLOW)

        if zero or (sign != overflow):
            self._jmp(target)

    def _ja(self, target):
        """Jump if above (carry clear and zero clear)"""
        carry = bool(self.registers['eflags'] & self.FLAG_CARRY)
        zero = bool(self.registers['eflags'] & self.FLAG_ZERO)

        if not carry and not zero:
            self._jmp(target)

    def _jae(self, target):
        """Jump if above or equal (carry clear)"""
        carry = bool(self.registers['eflags'] & self.FLAG_CARRY)

        if not carry:
            self._jmp(target)

    def _jb(self, target):
        """Jump if below (carry set)"""
        carry = bool(self.registers['eflags'] & self.FLAG_CARRY)

        if carry:
            self._jmp(target)

    def _jbe(self, target):
        """Jump if below or equal (carry set or zero set)"""
        carry = bool(self.registers['eflags'] & self.FLAG_CARRY)
        zero = bool(self.registers['eflags'] & self.FLAG_ZERO)

        if carry or zero:
            self._jmp(target)

    def _loop(self, target):
        """Decrement ECX and jump if ECX != 0"""
        # Decrement ECX
        self.registers['ecx'] = (self.registers['ecx'] - 1) & 0xFFFFFFFF

        # Jump if ECX != 0
        if self.registers['ecx'] != 0:
            self._jmp(target)

    def _push(self, src):
        """Push value onto stack"""
        value = self._get_value(src)
        if self.error:
            return

        # Decrement stack pointer
        self.registers['esp'] -= 4

        # Check stack overflow
        if self.registers['esp'] < 0:
            self.error = "Stack overflow"
            self.registers['esp'] += 4  # Restore stack pointer
            return

        # Write value to stack as 4 bytes (little-endian)
        for i in range(4):
            byte_val = (value >> (i * 8)) & 0xFF
            if not self._set_value(f"[{self.registers['esp'] + i}]", byte_val):
                # Restore stack pointer on error
                self.registers['esp'] += 4
                return

    def _pop(self, dest):
        """Pop value from stack"""
        # Check stack underflow
        if self.registers['esp'] >= self.memory_size - 4:
            self.error = "Stack underflow"
            return

        # Read 4 bytes from stack (little-endian)
        value = 0
        for i in range(4):
            byte_val = self._get_value(f"[{self.registers['esp'] + i}]")
            if self.error:
                return
            value |= byte_val << (i * 8)

        # Increment stack pointer
        self.registers['esp'] += 4

        # Store value to destination
        self._set_value(dest, value)

    def _call(self, target):
        """Call subroutine: push return address and jump to target"""
        # Get target address
        target_val = 0
        if target.lower() in self.labels:
            target_val = self.labels[target.lower()]
        else:
            target_val = self._get_value(target)
            if self.error:
                return

        # Save return address (current eip)
        self._push(str(self.registers['eip']))
        if self.error:
            return

        # Jump to target
        if 0 <= target_val < self.memory_size:
            self.registers['eip'] = target_val
        else:
            self.error = f"Call target out of bounds: {target_val}"

    def _ret(self):
        """Return from subroutine: pop return address and jump to it"""
        # Get return address from stack
        self._pop('eax')
        if self.error:
            return

        # Jump to return address
        self.registers['eip'] = self.registers['eax']

    def _interrupt(self, interrupt_num):
        """Handle software interrupt"""
        # Get interrupt number
        int_val = self._get_value(interrupt_num)
        if self.error:
            return

        # Handle specific interrupts
        if int_val == 0x80:  # System call (Linux-like)
            # Get syscall number from eax
            syscall = self.registers['eax']

            # Implement a few safe syscalls
            if syscall == 1:  # sys_exit
                # End program execution
                self.registers['eip'] = self.memory_size  # Jump to end of memory
            elif syscall == 4:  # sys_write
                # Simplified write syscall - doesn't actually print anything in the game
                # Just marks memory as accessed
                buf_ptr = self.registers['ecx']
                size = self.registers['edx']

                # Ensure buffer is within bounds
                if 0 <= buf_ptr < self.memory_size and 0 <= buf_ptr + size < self.memory_size:
                    # Mark as a successful write
                    self.registers['eax'] = size
                else:
                    self.registers['eax'] = 0xFFFFFFFF  # Error
            else:
                # Unimplemented syscall
                self.registers['eax'] = 0xFFFFFFFF  # Error

    def _update_flags(self, result, op1, op2, op_type):
        """Update CPU flags based on result"""
        flags = 0

        # Zero flag
        if result == 0:
            flags |= self.FLAG_ZERO

        # Sign flag
        if result & 0x80000000:
            flags |= self.FLAG_SIGN

        # Overflow flag (for add/sub)
        if op_type == 'add':
            # Overflow occurs when adding two numbers with the same sign
            # but the result has a different sign
            if ((op1 & 0x80000000) == (op2 & 0x80000000)) and ((result & 0x80000000) != (op1 & 0x80000000)):
                flags |= self.FLAG_OVERFLOW
        elif op_type == 'sub':
            # Overflow occurs in subtraction when the signs of the operands differ
            # and the sign of the result differs from the sign of the first operand
            if ((op1 & 0x80000000) != (op2 & 0x80000000)) and ((result & 0x80000000) != (op1 & 0x80000000)):
                flags |= self.FLAG_OVERFLOW
        elif op_type == 'neg':
            # Overflow occurs in negation when the operand is the most negative value
            if op1 == 0x80000000:
                flags |= self.FLAG_OVERFLOW

        # Carry flag
        if op_type == 'add':
            # Carry occurs when the result of unsigned addition exceeds 32 bits
            if (op1 + op2) > 0xFFFFFFFF:
                flags |= self.FLAG_CARRY
        elif op_type == 'sub':
            # Carry (borrow) occurs when op2 > op1 in unsigned subtraction
            if op2 > op1:
                flags |= self.FLAG_CARRY
        elif op_type == 'neg':
            # Carry is set for non-zero operand
            if op1 != 0:
                flags |= self.FLAG_CARRY

        self.registers['eflags'] = flags

    def dump_memory(self, start=0, length=64):
        """Dump a section of memory for debugging"""
        result = []
        for i in range(start, min(start + length, self.memory_size), 16):
            hex_vals = ' '.join(f'{self.memory[i+j]:02X}' for j in range(min(16, self.memory_size - i)))
            ascii_vals = ''.join(chr(self.memory[i+j]) if 32 <= self.memory[i+j] <= 126 else '.'
                               for j in range(min(16, self.memory_size - i)))
            result.append(f'{i:04X}: {hex_vals.ljust(48)} | {ascii_vals}')
        return '\n'.join(result)

    def dump_registers(self):
        """Dump all registers for debugging"""
        result = []
        for reg, val in self.registers.items():
            result.append(f'{reg.upper()}: {val:08X}')
        return '\n'.join(result)
