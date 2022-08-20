pub mod loader;
use std::env;


pub mod network2;
use network2::Network;
use ndarray::{Array, Array2};
use ndarray_rand::RandomExt;

fn main() -> Result<(), std::io::Error> {
    env::set_var("RUST_BACKTRACE", "1");
    let mut images = loader::MNistData::load("train")?;
    let test = loader::MNistData::load("t10k")?;
    let mut net = Network::new( &[784,16,16,10] );

    net.sgd(&mut images, 30, 10, 3, &test);
    Ok(())
}
