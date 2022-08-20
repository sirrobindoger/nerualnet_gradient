use core::num;

use ndarray_rand::{rand_distr::StandardNormal, rand::{seq::SliceRandom, thread_rng}};
use ndarray::{Array, Array2};
use ndarray_rand::RandomExt;

pub struct Network {
    num_layers: usize,
    biases: Vec<Array2<f64>>,
    weights: Vec<Array2<f64>>
}
#[derive(Clone)]
pub struct Data {
    pub image: Array2<f64>,
    pub label: Array2<f64>
}

impl Network {
    pub fn new(sizes: &[usize]) -> Network {
        let num_layers = sizes.len();
        let mut biases: Vec<Array2<f64>> = Vec::new();
        let mut weights: Vec<Array2<f64>> = Vec::new();
        for y in &sizes[1..] {
            biases.push(Array2::random((*y,1), StandardNormal));
        }
        for (x,y) in sizes[..num_layers-1].iter().zip(&sizes[1..]) {
            weights.push(Array::random((*y,*x), StandardNormal));
        }

        Network {
            num_layers,
            biases,
            weights,
        }
    }
    pub fn feedforward(&self, input : &Array2<f64>) -> Array2<f64> {
        let mut ret = input.to_owned();
        let it = self.weights.iter().zip(self.biases.iter());
        for (_i, (w, b)) in it.enumerate() {
            ret = Network::sigmoid( &(w.dot(&ret) + b) ); // set the activations to the sigmoid output of the previous
        }
        ret
    }

    pub fn feed_forward(&self, input : &mut Array2<f64>) {
        let it = self.weights.iter().zip(self.biases.iter());
        for (_i, (w, b)) in it.enumerate() {
            *input = Network::sigmoid( &(w.dot(input) + b) ); // set the activations to the sigmoid output of the previous
        }
    }

    pub fn sgd(
        &mut self, 
        training_data: &mut [Data], 
        epochs: usize, 
        mini_batch_size: usize, 
        eta: usize, 
        test_data: &[Data]
    ){
        let n_train = training_data.len();
        let n_test = test_data.len();

        for i in 0..epochs {
            let mut mini_batches: Vec<Vec<Data>> = Vec::new();
            training_data.shuffle( &mut thread_rng() );
            for j in (0..n_train).step_by(mini_batch_size) {
                mini_batches.push(
                    training_data[ j..j+mini_batch_size ].to_vec()
                );
            }
            for mini_batch in mini_batches {
                self.update_mini_batch(mini_batch, eta as f64);
            }

            println!("Evalutating");
            println!("Epoch {0}: {1} / {2}", i, self.evaluate(test_data), n_test);
        }
    }

    fn update_mini_batch(&mut self, mini_batch: Vec<Data>, eta: f64) {
        let mut nabla_b = Network::to_zeros(&self.biases);
        let mut nabla_w = Network::to_zeros(&self.weights);

        for Data {image, label} in &mini_batch { 
            let (delta_nabla_b, delta_nabla_w) = self.backprop(image, label);
            nabla_b = nabla_b.iter().zip(&delta_nabla_b).map(|(nb, dnb)| nb+dnb).collect();
            nabla_w = nabla_w.iter().zip(&delta_nabla_w).map(|(nw, dnw)| nw+dnw).collect();
        }
        let n_batch = mini_batch.len() as f64;
        self.weights = self.weights.iter().zip(&nabla_w).map(|(w, nw)| {
            w-(eta/n_batch)*nw
        }).collect();
        self.biases = self.biases.iter().zip(&nabla_b).map(|(b, nb)| {
            b-(eta/n_batch)*nb
        }).collect();


    }

    fn backprop(&self, x: &Array2<f64>, y: &Array2<f64>) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let mut nabla_b = Network::to_zeros(&self.biases);
        let mut nabla_w = Network::to_zeros(&self.weights);

        let mut activation = x.to_owned();
        let mut activations = vec![ x.to_owned() ];

        let mut zs: Vec<Array2<f64>> = Vec::new();
        // feed forward
        for (b,w) in self.biases.iter().zip(&self.weights) {
            let z = w.dot(&activation) + b;
            zs.push(z.to_owned());
            activation = Network::sigmoid(&z);
            activations.push(activation.to_owned());
        }
        //backwards pass
        let mut delta = Network::cost_derivative(activations.last().unwrap(), y) * 
            Network::sigmoid_prime(zs.last().unwrap());
        let n_nabla_b = nabla_b.len();
        let n_nabla_w = nabla_w.len();
        let n_activations = activations.len();
        //std::mem::replace(&mut nabla_b[ n_nabla_b - 1 ], delta);
        nabla_w[ n_nabla_w - 1 ] = delta.dot( 
            &activations[ n_activations - 2 ]
            .t() 
        ); 
        nabla_b[ n_nabla_b - 1 ] = delta.to_owned();
        for i in 2 ..self.num_layers {
            let z = zs[ zs.len() - i ].to_owned();
            let sp = Network::sigmoid_prime(&z);
            let n_nabla_b = nabla_b.len();
            let n_nabla_w = nabla_w.len();
            let n_activations = activations.len();
            delta = self.weights[ self.weights.len() - i + 1 ].t().dot(&delta) * sp;
            nabla_b[ n_nabla_b - i ] = delta.to_owned();
            nabla_w[ n_nabla_w - i ] = delta.dot( &activations[ n_activations - i - 1 ].t() );
            
        };
        (nabla_b, nabla_w)
    }

    fn evaluate(&self, test_data: &[Data]) -> usize {
        
        let test_results = test_data
            .iter()
            .map(|x| self.feedforward(&x.image.to_owned()))
            .map(|x| Network::argmax(&x))
            .collect::<Vec<usize>>();

        test_results
            .iter()
            .zip(test_data.iter())
            .map(|(x, y)| usize::from(*x == Network::argmax(&y.label) as usize))
            .sum()
    }

    fn argmax(a: &Array2<f64>) -> usize {
        //println!("Evalutating2");
        let mut ret = 0;
        //dbg!(&a.shape());
        for (i, el) in a.iter().enumerate() {
            if *el > a[[ret, 0]] {
                //dbg!(&a);
                ret = i;
            }
        }
        ret
    }

    fn sigmoid(z: &Array2<f64>) -> Array2<f64> {
        1.0 / (1.0 + (-z).mapv(f64::exp))
    }

    fn cost_derivative(output_activations: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
        //dbg!(output_activations.shape(), y.shape());
        output_activations.to_owned() - y
    }

    fn sigmoid_prime(z : &Array2<f64>) -> Array2<f64> {
        Network::sigmoid(z) * (1f64 - Network::sigmoid(z))
    }

    fn to_zeros(arr : &[Array2<f64>]) -> Vec<Array2<f64>> {
        arr.iter().map(|w| Array::zeros(w.raw_dim()) ).collect()
    }
}