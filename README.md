# Ising

Montecarlo algorithm of Ising model implemented in Rust and Python

## What is coded in Rust
Interface with Python via a class `Hamiltonian`.
- spin flip
- energy computation

## What is coded in Python
The neighbour and next neighbour lists are computed in Python then used to create a `Hamiltonian` instance.

## To run

- Install rust [here](https://www.rust-lang.org/en-US/install.html) or with `sudo apt-get install rustc cargo`
- Install python3 with `sudo apt-get install python3 python3-numpy python3-matplotlib`
- Install jupyther with `sudo apt-get install jupyter-notebook`
- Compile the rust code with the command `cargo build --release`
- Go on jupyter with `jupyter-notebook`
