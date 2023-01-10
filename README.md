# translator

My goal here is to build a Neural Machine Translation program from scratch, entirely in Rust.
For now, I'm starting with a simpler problem: building a feedforward neural network and training it to classify digits from the MNIST dataset. I got annoyed of manually computing derivatives so I am pausing `translator` while I build an autodifferentiation module called [`drift`](https://github.com/mmasque/drift). 

## Motivation
This is mostly a learning exercise where I hope to improve my Rust skills and internalise fundamental machine learning concepts. 

## References
- I'm implementing a feedforward network based on the seminal paper *Rumelhart, David E., Geoffrey E. Hinton, and Ronald J. Williams. "Learning representations by back-propagating errors."* You can find a pdf [here](https://www.nature.com/articles/323533a0.pdf?origin=ppub).
- For some background on Neural Machine Translation, see the [Wikipedia article](https://en.wikipedia.org/wiki/Neural_machine_translation). 
- I will likely implement a simplified version of the Google translation system, which is described in [this pre-print](https://arxiv.org/pdf/1609.08144.pdf).  
