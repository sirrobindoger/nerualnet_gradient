pub mod loader;
use loader::MNistData;

fn main() {
    MNistData::new(String::from("mnist\\train-images-idx3-ubyte.gz"));
}
