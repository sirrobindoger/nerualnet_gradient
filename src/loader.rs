use std::{fs::File, io::{BufReader, Read, Cursor}};

use flate2::bufread::GzDecoder;
use byteorder::{BigEndian, ReadBytesExt};
use ndarray::{Array2};

use crate::network2::Data;


pub struct MNistData {
    sizes: Vec<i32>,
    data: Vec<u8>,
}

pub struct MnistImage {
    pub image: Array2<f64>,
    pub classification: u8,
}

impl MNistData {
    /// # Panics
    /// 
    /// Will panic if there's no idx file
    /// # Errors
    /// 
    /// Will error if no idx file
    fn new(file: String) -> Result<MNistData, std::io::Error> {
        // loading & decompressing
        let buffer = BufReader::new( File::open(file)?);
        let mut contents: Vec<u8> = Vec::new();
        
        GzDecoder::new(buffer).read_to_end(&mut contents)?;
        // seeking magic number
        let mut reader = Cursor::new(&contents);

        let mut sizes: Vec<i32> = Vec::new();
        let mut data: Vec<u8> = Vec::new();

        match reader.read_i32::<BigEndian>()? {
            2049 => {
                sizes.push(reader.read_i32::<BigEndian>()?);
            }
            2051 => {
                sizes.push(reader.read_i32::<BigEndian>()?);
                sizes.push(reader.read_i32::<BigEndian>()?);
                sizes.push(reader.read_i32::<BigEndian>()?);
            }
            _ => panic!(),
        }

        reader.read_to_end(&mut data)?;

        Ok(MNistData{sizes, data})
    }
    /// # Errors
    /// 
    /// Errors if no model files
    ///  # Panics
    /// 
    /// Errors if no model files
    pub fn load(name: &str) -> Result<Vec<Data>, std::io::Error> {
        let images_data = &MNistData::new(format!("mnist\\{}-images-idx3-ubyte.gz", name))?;
        let labels = &MNistData::new(format!("mnist\\{}-labels-idx1-ubyte.gz", name))?;

        let mut images : Vec<Array2<f64>> = Vec::new();
        let image_shape = (images_data.sizes[1] * images_data.sizes[2]) as usize;
       
        for i in 0..images_data.sizes[0] as usize {
            let start = i * image_shape;
            let curr_image: Vec<f64> = images_data.data[start..start + image_shape]
                .iter()
                .map(|x| f64::from(*x) / 255.)
                .collect();
            images.push(Array2::from_shape_vec((image_shape, 1), curr_image).unwrap());
        }

        let classifications: Vec<u8> = labels.data.to_owned();

        let mut ret: Vec<Data> = Vec::new();
        
        for (image, classification) in images.into_iter().zip(classifications.into_iter()) {
            let mut label = Array2::zeros((10, 1));
            label[[classification as usize, 0]] = 1f64;
        
            ret.push(Data{image, label});
        }
        Ok(ret)

    }
}