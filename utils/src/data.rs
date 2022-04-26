use std::{path::{Path, PathBuf}, io, fs::{self, ReadDir}};
use anyhow::bail;
use thiserror::{Error,};

pub trait Dataset<X: Send, Y: Send>: Iterator<Item = (X, Y)> + Send {}

#[derive(Debug, Error)]
enum DataSetError{
    #[error("The two folder dont have the same size")]
    NotTheSameSize
}

pub struct Datafolder<'a, X: Send, Y: Send>{
    x_loader: &'a (dyn Fn(PathBuf) -> X + Send + Sync),
    y_loader: &'a (dyn Fn(PathBuf) -> Y + Send + Sync),
    iter_x: ReadDir,
    iter_y: ReadDir
}

impl<'a, X: Send, Y: Send> Datafolder<'a, X,Y>{
    pub fn from(path: &Path, x_folder:String, y_folder:String, x_loader: &'a (dyn Fn(PathBuf)-> X + Send + Sync), y_loader:  &'a (dyn Fn(PathBuf)-> Y + Send + Sync)) -> anyhow::Result<Self> {
        let x_path = path.join( x_folder);
        let y_path = path.join( y_folder);

        if ! x_path.exists(){
            bail!(io::Error::new(io::ErrorKind::NotFound, format!("{x_path:?} not found")));
        }
        if ! y_path.exists(){
            bail!(io::Error::new(io::ErrorKind::NotFound, format!("{y_path:?} not found")))
        }
        if ! x_path.is_dir(){
            bail!(io::Error::new(io::ErrorKind::NotFound, format!("{x_path:?} is not a directory")))
        }
        if ! y_path.is_dir(){
            bail!(io::Error::new(io::ErrorKind::NotFound, format!("{y_path:?} is not a directory")))
        }
        if fs::read_dir(x_path.as_path())?.count() != fs::read_dir(y_path.as_path())?.count() {
            bail!(DataSetError::NotTheSameSize);
        }
        let iter_x = fs::read_dir(x_path.as_path())?;
        let iter_y = fs::read_dir(y_path.as_path())?;
        Ok(Self{
            x_loader,
            y_loader,
            iter_x,
            iter_y
        })
    }
}

impl<'a, X: Send, Y: Send> Dataset<X, Y> for Datafolder<'a, X, Y> {

}

impl<'a, X: Send, Y: Send> Iterator for Datafolder<'a, X,Y> {
    type Item = (X,Y);

    fn next(&mut self) -> Option<Self::Item> {
        let y = self.iter_y.next();
        let x = self.iter_x.next();

        let (x,y) =  match (x,y){
            (Some(Ok(x)), Some(Ok(y))) => (x,y),
            _=> return None
        };

        Some(((self.x_loader)(x.path()), (self.y_loader)(y.path())))
    }
}

/*pub trait DataLoader<X, Y, BX, BY>: Iterator<Item = (BX, BY)>
    where BX: FromIterator<X>, BY: FromIterator<Y>{}


pub struct SingleThreadedDataLoader< X: Send, Y: Send, BX, BY> where 
    BX: FromIterator<X>, BY: FromIterator<Y>{
    batch_size:usize,
    iter: Box<dyn Iterator<Item = (BX, BY)>>, 
    phantom_data: PhantomData<(X,Y)>
}


impl< X: Send, Y: Send, BX, BY> SingleThreadedDataLoader< X, Y, BX, BY>
    where BX: FromIterator<X>, BY: FromIterator<Y>{
    
    pub fn new(dataset: Box<dyn Dataset<X,Y>>, batch_size:usize)->Self{
        let iter = dataset.chunks(batch_size).into_iter().map(|batch| multiunzip(batch));
        Self{ batch_size, iter:Box::new(iter), phantom_data: PhantomData }
    }
}

impl< X: Send, Y: Send, BX, BY> DataLoader<X, Y, BX, BY> for SingleThreadedDataLoader<X, Y, BX, BY>
    where BX: FromIterator<X>, BY: FromIterator<Y>{}

impl< X: Send, Y: Send, BX, BY> Iterator for SingleThreadedDataLoader<X, Y, BX, BY> 
    where BX: FromIterator<X>, BY: FromIterator<Y>{
    type Item = (BX, BY);

    fn next(&mut self) -> Option<Self::Item> {
        self.dataset.chunks(self.batch_size)
    }
} */