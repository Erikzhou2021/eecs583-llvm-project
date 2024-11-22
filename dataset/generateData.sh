extractData() {
    # for file in *.c; do
    for file in $(ls -1 . | grep \.c\$ | sort); do
    echo ${file}
    # compile to llvm IR
    ../build/bin/clang -emit-llvm -std=c89 -c ${file} -o temp.bc || continue
    # clang -emit-llvm -c ${file} -o temp.bc -O3
    # clang -c ${file} -o temp.bc

    # run the pass
    ../build/bin/llc --regalloc=basic temp.bc 2> ${file}.txt
    # ../build/bin/llc --regalloc=greedy temp.bc 2> ${file}.txt

    rm temp.bc
    rm temp.s
    done
}

removeFiles(){
    for file in $(ls -1 . | grep \.c.txt\$ | sort); do
    echo ${file}
    rm ${file}
    done
}

removeFiles
# extractData
