extractData() {
    # for file in *.c; do
    for file in $(ls -1 . | grep \.c\$ | sort); do
    # compile to llvm IR
    ../build/bin/clang -emit-llvm -std=c89 -c ${file} -O3 -o temp.bc || continue
    # clang -emit-llvm -c ${file} -o temp.bc -O3
    # clang -c ${file} -o temp.bc

    # run the pass
    echo ${file} > ${file}.txt
    # ../build/bin/llc -O3 temp.bc 2>> ${file}.txt
    ../build/bin/llc --regalloc=greedy temp.bc 2>> ${file}.txt

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

# removeFiles
extractData
