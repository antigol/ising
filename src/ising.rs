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

fn energy(py: Python, state: &PyObject, nn: &Vec<(f64, Vec<Vec<usize>>)>) -> PyResult<f64> {
    let state = PyBuffer::get(py, state)?;
    let state = State::from_pybuffer(py, &state)?;
    Ok(state.compute_energy(nn))
}

fn sweep(py: Python, state: &PyObject, temp: f64, nns: &Vec<(f64, Vec<Vec<usize>>)>) -> PyResult<f64> {
    let state_buffer = PyBuffer::get(py, state)?;
    let mut state = State::from_pybuffer(py, &state_buffer)?;
    let mut delta_energy = 0.0;
    py.allow_threads(|| { delta_energy = state.sweep(temp, nns); });
    state.copy_to_pybuffer(py, &state_buffer)?;
    Ok(delta_energy)
}

// defines the python module
py_module_initializer!(ising, initising, PyInit_ising, |py, m| {
    m.add(py, "__doc__", "This module is implemented in Rust.")?;
    m.add(py, "Hamiltonian", py.get_type::<Hamiltonian>())?;
    Ok(())
});
