use core::panic;
use std::collections::HashMap;

use proc_macro::TokenStream;
use quote::quote;
use syn::{
    parenthesized, parse::Parse, parse_macro_input, parse_quote, punctuated::Punctuated, Ident,
    ItemFn, LitInt, Stmt, Token,
};

extern crate proc_macro;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
enum TensorDimention {
    Wildcard,
    Literal(i64),
    Variable(Ident),
}

impl Parse for TensorDimention {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        if input.peek(Ident) {
            let id = input.parse::<Ident>()?;
            if id.to_string().as_str() == "_" {
                Ok(TensorDimention::Wildcard)
            } else {
                Ok(TensorDimention::Variable(id))
            }
        } else {
            let lit = input.parse::<LitInt>()?;
            Ok(TensorDimention::Literal(lit.base10_parse()?))
        }
    }
}

#[derive(Debug)]
struct TensorArg {
    ident: Ident,
    dims: Vec<TensorDimention>,
}

impl Parse for TensorArg {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ident = input.parse()?;
        let content;
        parenthesized!(content in input);
        let dims = Punctuated::<TensorDimention, Token![,]>::parse_terminated(&content)?;
        Ok(TensorArg {
            ident,
            dims: dims.into_iter().collect(),
        })
    }
}

impl TensorArg {
    fn get_dimention_assert(&self) -> Stmt {
        let ident = &self.ident;
        let idn = ident.to_string();
        let dims = self.dims.len();
        parse_quote!(
            assert!(#ident.size().len() == #dims, "Expected {} to have {} dimentions, got {} instead", #idn, #dims, #ident.size().len());
        )
    }
}

#[derive(Debug)]
struct Args {
    tensors: Vec<TensorArg>,
}

impl Parse for Args {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let tensors = Punctuated::<TensorArg, Token![,]>::parse_terminated(input)?;
        Ok(Args {
            tensors: tensors.into_iter().collect(),
        })
    }
}

#[proc_macro_attribute]
pub fn dims(args: TokenStream, input: TokenStream) -> TokenStream {
    let mut input = parse_macro_input!(input as ItemFn);
    let mut i = 0;
    // Parse the list of variables the user wanted to check.
    let args = parse_macro_input!(args as Args);

    // Checking the existance of the variables and adding dimentions check
    for arg in args.tensors.iter() {
        let id = &arg.ident;
        let farg = input.sig.inputs.iter().find(|arg| match arg {
            syn::FnArg::Receiver(_) => false,
            syn::FnArg::Typed(t) => match &*t.pat {
                syn::Pat::Ident(other) => &other.ident == id,
                _ => false,
            },
        });
        if farg.is_some() {
            input.block.stmts.insert(i, arg.get_dimention_assert());
            i += 1;
        } else {
            panic!("DimsError : {id} is not a parameter");
        }
    }

    // Building a symbole table
    let mut symbols_table = HashMap::new();
    for arg in args.tensors.iter() {
        for (index, dim) in arg.dims.iter().enumerate() {
            symbols_table
                .entry(dim.clone())
                .or_insert(vec![])
                .push((index, arg.ident.clone()))
        }
    }

    //Given the symbole table build the asserts that check the dims of the tensor
    for (k, v) in symbols_table.into_iter() {
        match k {
            TensorDimention::Wildcard => (),
            TensorDimention::Literal(lit) => {
                for (idx, id) in v {
                    let id_name = id.to_string();
                    let stmt: Stmt = parse_quote!(
                        assert!(#id.size()[#idx] == #lit, "Expected {} to have a size of {} for the dimention {}, got {} instead", #id_name, #lit, #idx, #id.size()[#idx]);
                    );
                    input.block.stmts.insert(i, stmt);
                    i += 1;
                }
            }
            TensorDimention::Variable(var_id) => {
                // If only one tensor use the variable no use to make asserts
                if v.len() < 2 {
                    continue;
                }

                let var_id_n = var_id.to_string();

                let mut it = v.into_iter();
                // picking an anchor before iterating regularly
                let (anc_idx, anc_id) = it.next().unwrap();

                for (idx, id) in it {
                    let stmt: Stmt = parse_quote!(
                        assert!(#anc_id.size()[#anc_idx] == #id.size()[#idx], "Size mismatch on tensors with the dimention {}", #var_id_n);
                    );
                    input.block.stmts.insert(i, stmt);
                    i += 1;
                }
            }
        }
    }

    // Hand the resulting function body back to the compiler.
    TokenStream::from(quote!(#input))
}
