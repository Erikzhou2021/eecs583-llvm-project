"""
Tokenization scheme:
physical registers: 0 - 1000
virtual registers: 1000 - 2000
basic block start / end 3000, 3001
basic block numbers: 3002 - 4000
op codes: 4000+
"""


def tokenize_inst(inst, only_reg=True):
    tokens = [] if only_reg else [] # might not need a special token for this idk
    opcode, registers = inst.split(":")
    opcode = int(opcode.split("|")[1])
    opcode += 4000
    if not only_reg: tokens.append(opcode)
    if registers != '':
        registers = registers.split(",")
        tokens += [int(r) for r in registers]
    # if not only_reg: tokens.append(-1)
    if opcode == 4019: return []
    return tokens


def tokenize_bb(bb, only_reg=True):
    tokens = [] if only_reg else [3000]
    instructions = bb.strip(" ").split(" ")
    for inst in instructions:
        tokens += tokenize_inst(inst, only_reg=only_reg)
    if not only_reg: tokens.append(3001)
    return tokens


def tokenized_format(input_file, state="post", only_reg=True):
    """
    1.  Converts the pre pass into a tokenized format.
        Format:
            <BB_START> BB#
            <INST> opcode r1 r2 ... </INST>
            <INST> opcode r3 r4 ... </INST>
            </BB_END>
    2.  Generates an output mapping virtual to physical registers from pre and post passes.
        Format:
            <MAP> BB#
            
            </MAP>

    Output: a list of lists, where each inner list is a basic block
    """
    with open(input_file, "r") as f:
        bb_tokens = []
        txt = f.read()
        pre, post = txt.split('~')
        
        p = pre if state == "pre" else post
        basic_blocks = p.strip().split('\n')
        for bb in basic_blocks:
            bb_tokens += tokenize_bb(bb, only_reg=only_reg)
    return bb_tokens
        

def print_tokens(bb_tokens):
    for bb in bb_tokens:
        for b in bb:
            print(b, end=" ")
            if b[:2] == "BB" or b == "</INST>":
                print()
        print('\n')

bb_tokens = tokenized_format("hw2correct1.c.txt")
print(bb_tokens)
# bb_tokens = tokenized_format("spill.c.txt")
# print_tokens(bb_tokens)
# print(bb_tokens)