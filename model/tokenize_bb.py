"""
Tokenization scheme:
physical registers: 0 - 1000
virtual registers: 1000 - 2000
basic block numbers: 3000 - 4000
op codes: 4000+
"""
import glob

def tokenize_inst(inst, only_reg=True, include_tags=False):
    tokens = []
    if inst == "":
        return tokens
    opcode, registers = inst.split(":")
    opcode = int(opcode.split("|")[1])
    if opcode == 19: return []
    opcode += 4000
    if not only_reg: tokens.append(opcode)
    if registers != '':
        registers = registers.split(",")
        tokens += [int(r) for r in registers]
    return tokens

def tokenize_bb(bb, only_reg=True, include_tags=True):
    tokens = []
    if include_tags:
        block_number, bb = bb.strip(" ").split("$")
        tokens.append(int(block_number))
    instructions = bb.strip(" ").split(" ")
    for inst in instructions:
        tokens += tokenize_inst(inst, only_reg=only_reg, include_tags=include_tags)
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
        allTokens = []
        allText = f.read()
        passes = allText.split('&')

        for txt in passes:
            txt = txt.strip()
            if txt == "":
                continue
            bb_tokens = []
            pre, post = txt.split('~')
            # post = txt.split('~')
            # p = post
            # print(p)
            p = pre if state == "pre" else post
            basic_blocks = p.strip().split('\n')
            for bb in basic_blocks:
                bb_tokens += tokenize_bb(bb, only_reg=only_reg)
            allTokens.append(bb_tokens)
    return allTokens
        

def print_tokens(bb_tokens):
    for bb in bb_tokens:
        for b in bb:
            print(b, end=" ")
            if b[:2] == "BB" or b == "</INST>":
                print()
        print('\n')


file_paths = sorted(list(glob.glob("../dataset/*.txt")))
with open("allDataInput.csv", "w") as f:
    for path in file_paths:
        tokens = tokenized_format(path, state="pre", only_reg=False)
        
        for bb_tokens in tokens:
            f.write(",".join([str(i) for i in bb_tokens]))
            f.write('\n')

with open("allDataLabels.csv", "w") as f:
    for path in file_paths:
        tokens = tokenized_format(path, state="post", only_reg=False)
        
        for bb_tokens in tokens:
            f.write(",".join([str(i) for i in bb_tokens]))
            f.write('\n')
    

# bb_tokens = tokenized_format("../hw2bench/hw2correct1.c.txt", only_reg=True)
# print(bb_tokens)
# bb_tokens = tokenized_format("../hw2bench/hw2correct1.c.txt", only_reg=False)
# print(bb_tokens)


# bb_tokens = tokenized_format("spill.c.txt")
# print_tokens(bb_tokens)
# print(bb_tokens)