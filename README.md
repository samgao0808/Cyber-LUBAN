# Lab 0: Environment Setup and Debugging

## Background

In this course, you are expected to have basic familiarity with Linux and common development tools, such as `git`, `vim`/`gedit`, shell commands, and package management.

These basic Linux skills will not be covered in detail during the lab. Please make good use of search engines and online documentation when needed.

------

## Objectives

In this lab, you need to complete two tasks:

1. **Set up the development environment**: code editing, compiling with NASM/GCC/make, running with QEMU, and debugging with GDB.
2. **Run through the debugging workflow**: boot an i386 Linux kernel using QEMU, connect to it remotely using GDB, set breakpoints, single-step instructions, and inspect registers.

We use an existing Linux kernel in this lab because the goal is to get familiar with the workflow of:

```text
QEMU boots a kernel + GDB remotely debugs it
```

Starting from Lab 1, the debugging target will be replaced by the OS code written by ourselves, but the debugging method will remain the same.

------

# Part 1: Environment Setup

## 1.1 Install a Linux Environment

All labs in this course will be conducted under Linux. The recommended system is:

```text
Ubuntu 22.04 Desktop
```

You only need to prepare a usable Linux environment. You may choose any of the following:

- VMware
- VirtualBox
- WSL2
- Hyper-V virtual machine
- Dual boot Linux installation

Other Linux distributions or Ubuntu versions are also acceptable, but some package names may be different.

When creating your Linux user account, please use a username that can help TAs identify you. For example:

```text
Ye Wenjie -> wjye
```

------

## 1.2 Install Basic Development Tools

Update the package list:

```shell
sudo apt update
```

Install C/C++ compilation tools:

```shell
sudo apt install binutils gcc make
```

Check whether GCC is installed successfully:

```shell
gcc -v
```

Install other required tools:

```shell
sudo apt install nasm qemu-system-x86 gdb libc6-dev-i386
```

Install dependencies required for compiling the Linux kernel:

```shell
sudo apt install bison flex libssl-dev libncurses-dev bc
```

You may also install VS Code and useful extensions for C/C++ and assembly programming.

------

## 1.3 Check Tool Versions

Check QEMU:

```shell
qemu-system-i386 --version
```

Check NASM:

```shell
nasm -v
```

NASM version **2.15 or above** is recommended. Ubuntu 22.04 usually provides NASM 2.15 by default, which is sufficient for this course.

------

# Part 2: Compile the Linux Kernel

## 2.1 Download the Kernel

Create a working directory:

```shell
mkdir ~/lab0
cd ~/lab0
```

Download Linux kernel version 5.10.258:

```shell
https://cdn.kernel.org/pub/linux/kernel/v5.x/linux-5.10.258.tar.xz
```

Extract the source code:

```shell
xz -d linux-5.10.258.tar.xz
tar -xvf linux-5.10.258.tar
cd linux-5.10.258
```

------

## 2.2 Configure the Kernel

Configure the kernel as an i386 kernel:

```shell
make i386_defconfig
make menuconfig
```

In the menu interface, go to:

```text
Kernel hacking
  -> Compile-time checks and compiler options
```

Enable:

```text
Compile the kernel with debug info
```

This option is necessary because GDB needs debugging symbols.

Save the configuration and exit.

------

## 2.3 Compile the Kernel

Compile the kernel:

```shell
make -j8
```

You may change `8` according to the number of CPU cores on your machine.

After compilation, check whether the following files are generated:

```text
arch/x86/boot/bzImage
vmlinux
```

Here:

- `bzImage` is the compressed kernel image used by QEMU.
- `vmlinux` contains debugging symbols used by GDB.

------

# Part 3: Boot the Kernel with QEMU and Debug with GDB

QEMU and GDB are commonly used together for OS debugging.

The usual workflow is:

```text
Terminal 1: run QEMU
Terminal 2: run GDB and connect to QEMU
```

All commands below are executed under:

```shell
cd ~/lab0
```

------

## 3.1 Start QEMU

In the first terminal, run:

```shell
qemu-system-i386 \
  -kernel linux-5.10.x/arch/x86/boot/bzImage \
  -s -S \
  -append "console=ttyS0" \
  -nographic
```

Explanation of important options:

| Option                    | Meaning                                           |
| ------------------------- | ------------------------------------------------- |
| `-s`                      | Open a GDB server on port `1234`                  |
| `-S`                      | Pause the CPU at startup and wait for GDB         |
| `-nographic`              | Use terminal output instead of a graphical window |
| `-append "console=ttyS0"` | Redirect kernel output to the serial console      |

At this point, QEMU may show no output. This is normal because it is waiting for GDB.

------

## 3.2 Connect with GDB

Open a second terminal and run:

```shell
cd ~/lab0
gdb
```

In GDB, load the symbol table:

```shell
file linux-5.10.258/vmlinux
```

Connect to QEMU:

```shell
target remote:1234
```

Set a breakpoint at the kernel entry function:

```shell
break start_kernel
```

Continue execution:

```shell
c
```

The kernel should stop at `start_kernel`.

This means you have successfully completed the basic QEMU + GDB debugging workflow.

------

## 3.3 Practice Basic GDB Commands (Optional)

After stopping at `start_kernel`, try the following commands:

| Command          | Meaning                                                  |
| ---------------- | -------------------------------------------------------- |
| `b <function>`   | Set a breakpoint at a function                           |
| `b *0x7c00`      | Set a breakpoint at an address                           |
| `c`              | Continue execution                                       |
| `si`             | Execute one assembly instruction and step into functions |
| `ni`             | Execute one assembly instruction and step over functions |
| `s`              | Execute one C statement and step into functions          |
| `n`              | Execute one C statement and step over functions          |
| `info registers` | Show all CPU registers                                   |
| `x/10i $pc`      | Disassemble 10 instructions from the current PC          |
| `x/12xw $esp`    | Show 12 words from the top of the stack                  |
| `layout src`     | Open the source-code view                                |
| `layout asm`     | Open the assembly view                                   |
| `layout regs`    | Open the register view                                   |

You do not need to fully understand every command now. The goal is to become familiar with the basic debugging interaction.

------

# Expected Result

At the end of this lab, you should have:

1. A working Linux development environment.
2. NASM, GCC, make, QEMU, and GDB installed.
3. A successfully compiled i386 Linux kernel.
4. A successful remote debugging session between QEMU and GDB.
5. Experience with setting breakpoints, continuing execution, single-stepping, and inspecting registers.

In later labs, we will replace the Linux kernel with our own OS code, but the debugging workflow will remain the same.