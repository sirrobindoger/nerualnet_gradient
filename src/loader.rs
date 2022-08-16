use std::{fs::File, io::{BufReader, Read, Cursor}};

use flate2::bufread::GzDecoder;
use byteorder::{BigEndian, ReadBytesExt};


pub struct MNistData {
    sizes: Vec<i32>,
    data: Vec<u8>,
}

impl MNistData {
    /// # Panics
    /// 
    /// Will panic if there's no idx file
    /// # Errors
    /// 
    /// Will error if no idx file
    pub fn new(file: String) -> Result<MNistData, std::io::Error> {
        // loading & decompressing
        let buffer = BufReader::new( File::open(file)?);
        let mut contents: Vec<u8> = Vec::new();
        
        GzDecoder::new(buffer).read_to_end(&mut contents)?;
        // seeking magic number
        let mut reader = Cursor::new(&contents);

        let mut sizes: Vec<i32> = Vec::new();
        let mut data: Vec<u8> = Vec::new();

        assert_eq!(reader.read_i32::<BigEndian>()? , 2051); // check if we're using correct IDX

        for _ in 0..2 {
            sizes.push(reader.read_i32::<BigEndian>()?);
        };
        
        reader.read_to_end(&mut data)?;

        Ok(MNistData{sizes, data})
    }
}