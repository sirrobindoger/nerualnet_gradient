use std::{fs::File, io::{BufReader, Read, Cursor, BufRead}};

use flate2::bufread::GzDecoder;
use byteorder::{BigEndian, ReadBytesExt};


pub struct MNistData {

}

impl MNistData {
    pub fn new(file: String) -> Result<MNistData, std::io::Error> {
        // loading & decompressing
        let mut f = File::open(file)?;
        let buffer = BufReader::new(f);
        let mut gz = GzDecoder::new(buffer);
        // putting decompressed buffer into vec
        let mut contents: Vec<u8> = Vec::new();
        gz.read_to_end(&mut contents)?;
        // seeking magic number
        let mut cursor = Cursor::new(&contents);
        let magic_number = cursor.read_i32::<BigEndian>()?;

        let mut dims: Vec<i32> = Vec::new();
        let mut data: Vec<u8> = Vec::new();
        match magic_number {
            2051 => {
                dims.push(cursor.read_i32::<BigEndian>()?);
                dims.push(cursor.read_i32::<BigEndian>()?);
                dims.push(cursor.read_i32::<BigEndian>()?);
            }
            _ => panic!(),
        }
        
        cursor.read_to_end(&mut data)?;

        Ok(MNistData{})
    }
}