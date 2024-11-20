spills = 0
loads = 0

def tokenize_inst(inst):
    global spills, loads
    tokens = ["<INST>"]
    opcode, registers = inst.split(":")
    opcode = opcode.split("|")[1]
    tokens.append(opcode)
    if registers != '':
        registers = registers.split(",")
        tokens += registers
    tokens.append("</INST>")
    if opcode == "2369": spills += 1
    if opcode == "2361": loads += 1
    return tokens


def tokenize_inst_post(inst):
    global spills, loads
    tokens = ["<INST>"]
    opcode, registers = inst.split(":")
    opcode = opcode.split("|")[1]
    tokens.append(opcode)
    if registers != '':
        registers = registers.split(",")
        tokens += registers
    tokens.append("</INST>")
    if opcode == "2369": spills += 1
    if opcode == "2361": loads += 1
    if opcode == "19": return []
    return tokens


def tokenize_bb(bb, i):
    tokens = ["<BB_START>", f"BB{i}"]
    instructions = bb.strip(" ").split(" ")
    for inst in instructions:
        tokens += tokenize_inst(inst)
    tokens.append("<BB_END>")
    return tokens

def tokenize_bb_post(bb, i):
    tokens = ["<BB_START>", f"BB{i}"]
    instructions = bb.strip(" ").split(" ")
    for inst in instructions:
        tokens += tokenize_inst_post(inst)
    tokens.append("<BB_END>")
    return tokens


def tokenized_format(input_file):
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
        bb_tokens_pre = []
        bb_tokens_post = []
        txt = f.read()
        pre, post = txt.split('~')
        
        # pre-reg-alloc
        basic_blocks = pre.strip().split('\n')
        for i, bb in enumerate(basic_blocks):
            bb_tokens_pre.append(tokenize_bb_post(bb, i))
        
        # post-reg-alloc
        basic_blocks = post.strip().split('\n')
        for i, bb in enumerate(basic_blocks):
            bb_tokens_post.append(tokenize_bb(bb, i))
    return bb_tokens_pre, bb_tokens_post
        

def print_tokens(bb_tokens):
    for bb in bb_tokens:
        for b in bb:
            print(b, end=" ")
            if b[:2] == "BB" or b == "</INST>":
                print()
        print('\n')

# bb_tokens = tokenized_format("hw2correct1.c.txt")
bb_tokens_pre, bb_tokens_post = tokenized_format("spill.c.txt")
# print_tokens(bb_tokens)
# print(bb_tokens)