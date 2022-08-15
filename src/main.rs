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

    fn sigmoid_prime(z : &Array2<f64>) -> Array2<f64> {
        Network::sigmoid(z) * (1f64 - Network::sigmoid(z))
    }

    fn feed_forward(&self, input : &mut Array2<f64>) {
        let it = self.weights.iter().zip(self.bias.iter());
        for (_i, (w, b)) in it.enumerate() {
            *input = Network::sigmoid( &(w.dot(input) + b) ); // set the activations to the sigmoid output of the previous
        }
    }
    fn update_mini_batch(&mut self, batch: &Vec<(Array2<f64>, Array2<f64>)>, eta : f64) {
        // Update the network's weights and biases by using GD w/ backpropagation
        let mut nabla_w: Vec<Array2<f64>> =  self.weights.iter().map(|w| Array::zeros(w.raw_dim()) ).collect();
        let mut nabla_b: Vec<Array2<f64>> = self.bias.iter().map(|b| Array::zeros(b.raw_dim()) ).collect();
        for(inputs, outputs) in batch.iter() {
            let (d_nabla_b, d_nabla_w) = self.backprop(&inputs, &outputs); 
            for (nb, dnb) in nabla_b.iter_mut().zip(d_nabla_b.iter()) {
                *nb += dnb;
            }
            for (nb, dnb) in nabla_w.iter_mut().zip(d_nabla_w.iter()) {
                *nb += dnb;
            }
        };
        let batchlen = batch.len() as f64;
        for (w, nw) in self.weights.iter_mut().zip(nabla_w.iter()) {
            *w -= &nw.mapv(|x| x * eta / batchlen);
        }
        for (b, nb) in self.bias.iter_mut().zip(nabla_b.iter()) {
            *b -= &nb.mapv(|x| x * eta / batchlen);
        }
    }
    fn stochastic_gradient_decent(
        &mut self, 
        training_data: &mut Vec<(Array2<f64>, Array2<f64>)>,
        epochs: u8, 
        mini_batch_size: usize, 
        eta: f64,
    ) {
        let n = training_data.len();
        for i in 1..epochs {
            training_data.shuffle(&mut thread_rng());
            let mut mini_batches: Vec<_> = Vec::new();
            for j in (0..n).step_by(mini_batch_size) {
                mini_batches.push( 
                    training_data[j .. j+mini_batch_size ].to_vec()
                );
            }
            mini_batches.iter().map(|batch: &Vec<_>| self.update_mini_batch(batch, eta));
            println!("Epoch {0} complete", i);
        }
    }

    fn backprop(&self, x: &Array2<f64>, y: &Array2<f64>) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let mut nabla_w: Vec<Array2<f64>> =  self.weights.iter().map(|w| Array::zeros(w.raw_dim()) ).collect();
        let mut nabla_b: Vec<Array2<f64>> = self.bias.iter().map(|b| Array::zeros(b.raw_dim()) ).collect();
        let mut activation = x.clone();
        let mut activations = vec![activation.clone()];
        let mut zs: Vec<_> = vec![];
        for (b,w) in self.bias.iter().zip(&self.weights) {
            let z = w.dot(&activation) + b;
            activation = Network::sigmoid(&z);
            activations.push(activation.clone());
            zs.push(z);
        }
        // backward pass!
        let mut delta = self.cost_derivative(&activations.last().unwrap(), y) * 
            Network::sigmoid_prime(zs.last().unwrap());
        let n_nabla_b = nabla_b.len();
        let n_nabla_w = nabla_w.len();
        let n_activations = activations.len();
        //std::mem::replace(&mut nabla_b[ n_nabla_b - 1 ], delta);
        nabla_w[ n_nabla_w - 1 ] = delta.dot( 
            &activations[ n_activations - 2 ]
            .t() 
        ); 
        nabla_b[ n_nabla_b - 1 ] = delta.clone();
        /*# Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.s*/
        for i in 2 ..self.num_layers {
            let z = zs[ zs.len() - i ].clone();
            let sp = Network::sigmoid_prime(&z);
            let n_nabla_b = nabla_b.len();
            let n_nabla_w = nabla_w.len();
            let n_activations = activations.len();
            delta = self.weights[ self.weights.len() - i + 1 ].t().dot(&delta) * sp;
            nabla_b[ n_nabla_b - i ] = delta.clone();
            nabla_w[ n_nabla_w - i ] = delta.dot( &activations[ n_activations - i - 1 ].t() )
            
        };
        return (nabla_b, nabla_w)
    }

    fn cost_derivative(&self, output_activations: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
        output_activations.clone() - y
    }
    
}




fn main() {
    
}
