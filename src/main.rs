use core::num;

use ndarray_rand::{rand_distr::StandardNormal, rand::{seq::SliceRandom, thread_rng}};
use ndarray::{Array, Array2, ArrayBase};
use ndarray_rand::RandomExt;

#[derive(Debug)]
struct Network {
    num_layers: usize,
    sizes: Vec<usize>,
    bias: Vec<Array2<f64>>,
    weights: Vec<Array2<f64>>
}
#[derive(Clone)]
struct TrainingData {
    inputs: Array2<f64>,
    outputs: Array2<f64>
}

impl Network {
    fn new (sizes: &[usize]) -> Network {
        let num_layers = sizes.len();
        let mut biases: Vec<Array2<f64>> = Vec::new();
        let mut weights: Vec<Array2<f64>> = Vec::new();
        for i in 1..num_layers {
            biases.push(
                Array::random((sizes[i], 1), StandardNormal)
            );
            weights.push(
                Array::random((sizes[i], sizes[i-1]), StandardNormal)
            );
        }
        Network {
            num_layers: num_layers,
            sizes: sizes.to_owned(),
            bias: biases,
            weights: weights,
        }
    }

    fn sigmoid(z: &Array2<f64>) -> Array2<f64> {
        z.mapv(|a| 1.0/(1.0+-a.exp()))
    }

    fn feed_forward(&self, input : &mut Array2<f64>) {
        let it = self.weights.iter().zip(self.bias.iter());
        for (_i, (w, b)) in it.enumerate() {
            *input = Network::sigmoid( &(w.dot(input) + b) ); // set the activations to the sigmoid output of the previous
        }
    }
    fn update_mini_batch(&self, batch: &Vec<TrainingData>, eta : &usize) {
        // Update the network's weights and biases by using GD w/ backpropagation
        let mut nabla_w: Vec<Array2<f64>> =  self.weights.iter().map(|w| Array::zeros(w.raw_dim()) ).collect();
        let mut nabla_b: Vec<Array2<f64>> = self.bias.iter().map(|b| Array::zeros(b.raw_dim()) ).collect();
        unimplemented!()
    }
    fn stochastic_gradient_decent(
        &self, 
        training_data: &mut Vec<TrainingData>,
        epochs: u8, 
        mini_batch_size: usize, 
        eta: usize,
    ) {
        let n = training_data.len();
        for i in 1..epochs {
            training_data.shuffle(&mut thread_rng());
            let mut mini_batches: Vec<Vec<TrainingData>> = Vec::new();
            for j in (0..n).step_by(mini_batch_size) {
                mini_batches.push( 
                    training_data[j .. j+mini_batch_size ].to_vec()
                );
            }
            mini_batches.iter().map(|batch| self.update_mini_batch(&batch, &eta));
            println!("Epoch {0} complete", i);
        }
    }
}

fn main() {
    let net = Network::new( &[728,16,16,10] );
    let mut arr : Array2<f64> = Array::random((728, 1), StandardNormal);
    net.feed_forward(&mut arr);

    println!("{:?}", net);

}
