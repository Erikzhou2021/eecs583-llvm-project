extractData() {
    for file in *.c; do
    echo ${file}
    # compile to llvm IR
    clang -emit-llvm -c ${file} -o temp.bc

    # run the pass
    ../build/bin/llc --regalloc=basic temp.bc 2> ${file}.txt

    rm temp.bc
    rm temp.s
    done
}

extractData