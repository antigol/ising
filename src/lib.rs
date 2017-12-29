extern crate rand;

#[macro_use]
extern crate cpython;

use cpython::{PyResult, Python, PyObject};
use cpython::buffer::PyBuffer;

mod state;
use state::State;

py_class!(class Hamiltonian |py| {
    data nns: Vec<(f64, Vec<Vec<usize>>)>;

    def __new__(_cls, nns: Vec<(f64, Vec<Vec<usize>>)>) -> PyResult<Hamiltonian> {
        // let max_nn = nn.iter().map(|x| x.1.iter().map(|x| x.len()).max().unwrap()).collect();
        Hamiltonian::create_instance(py, nns)
    }
    def energy(&self, state: &PyObject) -> PyResult<f64> {
        energy(py, state, self.nns(py))
    }
    def sweep(&self, state: &PyObject, temp: f64) -> PyResult<f64> {
        sweep(py, state, temp, self.nns(py))
    }
});

fn energy(py: Python, state: &PyObject, nns: &Vec<(f64, Vec<Vec<usize>>)>) -> PyResult<f64> {
    let state_buffer = PyBuffer::get(py, state)?;

    // Python -> Rust : load state content into a State object
    let state = State::from_pybuffer(py, &state_buffer)?;

    Ok(state.compute_energy(nns))
}

fn sweep(py: Python, state: &PyObject, temp: f64, nns: &Vec<(f64, Vec<Vec<usize>>)>) -> PyResult<f64> {
    let state_buffer = PyBuffer::get(py, state)?;

    // Python -> Rust : load state content into a State object
    let mut state = State::from_pybuffer(py, &state_buffer)?;

    // perform the sweep in a manner that allow the multiprocessing
    let delta_energy = py.allow_threads(|| { state.sweep(temp, nns) });

    // Python <- Rust : put the new state into the buffer
    state.copy_to_pybuffer(py, &state_buffer)?;

    Ok(delta_energy)
}

// defines the python module
py_module_initializer!(ising, initising, PyInit_ising, |py, m| {
    m.add(py, "__doc__", "This module is implemented in Rust.")?;
    m.add(py, "Hamiltonian", py.get_type::<Hamiltonian>())?;
    Ok(())
});
