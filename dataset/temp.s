	.text
	.file	"20000703-1.c"
	.globl	foo                             # -- Begin function foo
	.p2align	4, 0x90
	.type	foo,@function
foo:                                    # @foo
	.cfi_startproc
# %bb.0:
	movb	$99, 19(%rdi)
	movw	$25185, 17(%rdi)                # imm = 0x6261
	movl	%esi, 20(%rdi)
	movl	%edx, 24(%rdi)
	retq
.Lfunc_end0:
	.size	foo, .Lfunc_end0-foo
	.cfi_endproc
                                        # -- End function
	.globl	bar                             # -- Begin function bar
	.p2align	4, 0x90
	.type	bar,@function
bar:                                    # @bar
	.cfi_startproc
# %bb.0:
	movups	.L.str.1(%rip), %xmm0
	movups	%xmm0, (%rdi)
	movb	$54, 16(%rdi)
	movw	$25185, 17(%rdi)                # imm = 0x6261
	movb	$99, 19(%rdi)
	movl	%esi, 20(%rdi)
	movl	%edx, 24(%rdi)
	retq
.Lfunc_end1:
	.size	bar, .Lfunc_end1-bar
	.cfi_endproc
                                        # -- End function
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:
	pushq	%rax
	.cfi_def_cfa_offset 16
	xorl	%edi, %edi
	callq	exit@PLT
.Lfunc_end2:
	.size	main, .Lfunc_end2-main
	.cfi_endproc
                                        # -- End function
	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"abc"
	.size	.L.str, 4

	.type	.L.str.1,@object                # @.str.1
.L.str.1:
	.asciz	"01234567890123456"
	.size	.L.str.1, 18

	.ident	"clang version 18.1.8 (git@github.com:Erikzhou2021/eecs583-llvm-project 92c53597067d44807e6e7207bd3df46f76f88330)"
	.section	".note.GNU-stack","",@progbits
