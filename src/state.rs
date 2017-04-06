use rand;
use rand::Rng;
use rand::distributions::{IndependentSample, Range};
use cpython::{Python, PyResult, PyErr};
use cpython::exc::TypeError;
use cpython::buffer::PyBuffer;

// A State contains the spins and the size of the lattice
pub struct State {
    spins: Vec<i32>,
}

#[allow(dead_code)]
impl State {
    // constructor of a new State from a PyBuffer
    pub fn from_pybuffer(py: Python, buffer: &PyBuffer) -> PyResult<State> {
        if buffer.dimensions() != 1 {
            return Err(PyErr::new::<TypeError, _>(py, "Not rank 1"));;
        }
        Ok(State { spins: buffer.to_vec(py)? })
    }

    pub fn copy_to_pybuffer(&self, py: Python, buffer: &PyBuffer) -> PyResult<()> {
        buffer.copy_from_slice(py, &self.spins)
    }

    pub fn compute_energy(&self, nns: &Vec<(f64, Vec<Vec<usize>>)>) -> f64 {
        let mut energy = 0f64;
        for i in 0..self.spins.len() {
            let mut sum = 0f64;
            for &(j, ref nn) in nns.iter() {
                for &k in nn[i].iter() {
                    sum += j * self.spins[k] as f64;
                }
            }
            energy += self.spins[i] as f64 * sum;
        }
        // feromagnetic
        // energy = - sum J s_i s_j
        -energy / 2.0
    }

    pub fn sweep(&mut self, temp: f64, nns: &Vec<(f64, Vec<Vec<usize>>)>) -> f64 {
        let mut delta_energy = 0f64;

        for _ in 0..self.spins.len() {
            delta_energy += self.try_flip(temp, nns);
        }

        delta_energy
    }

    fn try_flip(&mut self, temp: f64, nns: &Vec<(f64, Vec<Vec<usize>>)>) -> f64 {
        let mut rng = rand::thread_rng();

        let i = Range::new(0, self.spins.len()).ind_sample(&mut rng);

        // feromagnetic
        // energy = - sum_<ik> J s_i s_k
        // d_e =  energy_new(s_i flipped)    -  energy_old
        // d_e = (- (-s_i) * sum_<ik> J s_k) - (- s_i * sum_<ik> J s_k)
        // d_e = 2 * s_i * sum_<ik> J s_k

        let mut nei = 0f64;
        for &(j, ref nn) in nns.iter() {
            for &k in nn[i].iter() {
                nei += j * self.spins[k] as f64;
            }
        }

        let d_e = 2.0 * self.spins[i] as f64 * nei;

        if rng.next_f64() < f64::exp(- d_e / temp) {
            self.spins[i] = -self.spins[i];
            // energy_new = energy_old + d_e
            d_e
        } else {
            0.0
        }
    }
}
