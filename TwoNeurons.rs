use ndarray::concatenate;
use ndarray::prelude::*;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{self, Write, Read};
use std::path::Path;
use ndarray_rand::rand_distr::{Normal, Distribution};
use std::process::exit;
use std::collections::HashMap;
use rayon::prelude::*;

#[derive(Serialize, Deserialize, Debug)]
struct NeuralNetwork {
    weights: Vec<Array2<f64>>,
    biases: Vec<Array1<f64>>,
    neuron_counts: Vec<usize>,
    problem: bool, // True only if coherence error
}

impl NeuralNetwork {
    fn new(neuron_counts: Vec<usize>) -> NeuralNetwork {
        let mut rng: StdRng = StdRng::from_entropy();
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        /*for i in 0..neuron_counts.len() - 1 {
            let weight = Array2::<f64>::random_using(
                (neuron_counts[i + 1], neuron_counts[i]),
                Uniform::new(-0.5, 0.5),
                &mut rng,
            );
            let bias = Array1::<f64>::random_using(
                neuron_counts[i + 1],
                Uniform::new(-0.1, 0.1),
                &mut rng,
            );
            weights.push(weight);
            biases.push(bias);
        }*/

        for i in 0..neuron_counts.len() - 1 {
            let std_dev = (2.0 / neuron_counts[i] as f64).sqrt();
            let normal = match Normal::new(0.0, std_dev) {
                Ok(dist) => dist,
                Err(err) => {
                    eprintln!("Error creating normal distribution: {}", err);
                    exit(1);
                }
            };
            let weight = Array2::<f64>::random_using(
                (neuron_counts[i + 1], neuron_counts[i]),
                normal,
                &mut rng,
            );
            let bias = Array1::<f64>::zeros(neuron_counts[i + 1]);
            weights.push(weight);
            biases.push(bias);
        }

        NeuralNetwork {
            weights,
            biases,
            neuron_counts,
            problem: false,
        }
    }

    fn relu(x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|x| x.max(0.0))
    }

    fn relu_derivative(x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }

    fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut activation = input.clone();
        self.weights.iter().zip(&self.biases).enumerate().for_each(|(i, (weight, bias))| {
            let z = weight.dot(&activation) + bias;
            activation = if i == self.weights.len() - 1 {
                Self::sigmoid(&z)
            } else {
                Self::relu(&z)
            };
        });
        activation
    }

    fn train(&mut self, inputs: &Array2<f64>, outputs: &Array2<f64>, epochs: usize, learning_rate: f64, error_threshold: f64, decay_rate: f64) {
        self.problem = false;
        let initial_learning_rate = learning_rate;
        let mut decayed_learning_rate = initial_learning_rate;
    
        for epoch in 0..epochs {
            // Accumulate gradients in parallel
            let gradients: Vec<_> = inputs.outer_iter()
                .zip(outputs.outer_iter())
                .par_bridge()
                .map(|(input, output)| {
                    let input = input.to_owned();
                    let output = output.to_owned();
                    let mut activations = vec![input.clone()];
                    let mut zs = Vec::new();
    
                    // Forward pass
                    let mut activation = input;
                    for (i, (weight, bias)) in self.weights.iter().zip(&self.biases).enumerate() {
                        let z = weight.dot(&activation) + bias;
                        zs.push(z.clone());
                        activation = if i == self.weights.len() - 1 {
                            Self::sigmoid(&z)
                        } else {
                            Self::relu(&z)
                        };
                        activations.push(activation.clone());
                    }
    
                    // Backward pass
                    let mut nabla_w: Vec<Array2<f64>> = self.weights.iter().map(|w| Array2::zeros(w.raw_dim())).collect();
                    let mut nabla_b: Vec<Array1<f64>> = self.biases.iter().map(|b| Array1::zeros(b.raw_dim())).collect();
    
                    let z_last = &zs[zs.len() - 1];
                    let a_last = activations.last().unwrap();
                    let sigmoid_prime = Self::sigmoid_derivative(z_last);
                    let mut delta = (a_last - &output) * sigmoid_prime;
    
                    for i in (0..self.weights.len()).rev() {
                        if i != self.weights.len() - 1 {
                            delta = delta * Self::relu_derivative(&zs[i]);
                        }
    
                        let prev_activation = &activations[i];
                        nabla_w[i] = delta.view().insert_axis(Axis(1))
                            .dot(&prev_activation.view().insert_axis(Axis(0)));
                        nabla_b[i] = delta.clone();
                        delta = self.weights[i].t().dot(&delta);
                    }
    
                    (nabla_w, nabla_b)
                }).collect();
    
            // Aggregate and apply gradients
            for i in 0..self.weights.len() {
                let mut total_w = Array2::zeros(self.weights[i].dim());
                let mut total_b = Array1::zeros(self.biases[i].dim());
    
                for (grad_w, grad_b) in &gradients {
                    total_w = total_w.clone() + &grad_w[i];
                    total_b = total_b.clone() + &grad_b[i];
                }
    
                let batch_size = gradients.len() as f64;
                self.weights[i] = self.weights[i].clone() - (decayed_learning_rate / batch_size) * total_w;
                self.biases[i] = self.biases[i].clone() - (decayed_learning_rate / batch_size) * total_b;
            }
    
            let error = self.evaluate(inputs, outputs);
            if error <= error_threshold {
                self.problem = false;
                break;
            }
    
            if error > 1.0 {
                self.problem = true;
                continue;
            }
    
            if error > error_threshold {
                let last_hidden_layer = self.neuron_counts.len() - 3;
                self.add_neuron_to_layer(last_hidden_layer);
            }
    
            decayed_learning_rate = exponential_decay(epoch, initial_learning_rate, decay_rate);
        }
    }

    fn add_neuron_to_layer(&mut self, layer_index: usize) {
        assert!(
            layer_index < self.weights.len() - 1,
            "Cannot add neurons to output layer."
        );

        let new_bias = Array1::<f64>::zeros(1);
        self.biases[layer_index] =
            concatenate![Axis(0), self.biases[layer_index].clone(), new_bias];

        let input_size = self.weights[layer_index].dim().1;
        let new_weights = Array1::random(input_size, Uniform::new(-0.5, 0.5));
        self.weights[layer_index]
            .push_row(new_weights.view())
            .unwrap();

        let output_size = self.weights[layer_index + 1].dim().0;
        let new_out_weights = Array1::random(output_size, Uniform::new(-0.5, 0.5));
        self.weights[layer_index + 1]
            .push_column(new_out_weights.view())
            .unwrap();

        self.neuron_counts[layer_index + 1] += 1;
    }

    fn evaluate(&self, inputs: &Array2<f64>, outputs: &Array2<f64>) -> f64 {
        inputs.outer_iter().zip(outputs.outer_iter()).par_bridge().map(|(input, output)| {
            let prediction = self.forward(&input.to_owned());
            (&prediction - &output).mapv(|x| x.powi(2)).sum()
        }).sum::<f64>() / (inputs.nrows() as f64 * outputs.ncols() as f64)
    }
    

    // Save the neural network to a file
    fn save_to_file(&self, filename: &str) -> io::Result<()> {
        let file = File::create(filename)?;
        serde_json::to_writer(file, &self)?;
        Ok(())
    }

    // Load a neural network from a file
    fn load_from_file(filename: &str) -> io::Result<NeuralNetwork> {
        let file = File::open(filename)?;
        let nn: NeuralNetwork = serde_json::from_reader(file)?;
        Ok(nn)
    }

    fn sigmoid(x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }

    fn sigmoid_derivative(x: &Array1<f64>) -> Array1<f64> {
        let clipped = x.mapv(|v| v.max(-10.0).min(10.0));
        let s = Self::sigmoid(&clipped);
        &s * &(1.0 - &s)
    }
    
}

fn exponential_decay(epoch: usize, initial_lr: f64, decay_rate: f64) -> f64 {
    initial_lr * (decay_rate).powi(epoch as i32)
}

fn main() {
    let inputs = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];

    // A single array to hold the outputs for all gates
    let outputs = array![
        [0.01, 0.01, 0.01, 0.99, 0.99, 0.99], // input: [0, 0]
        [0.01, 0.99, 0.99, 0.01, 0.99, 0.01], // input: [0, 1]
        [0.01, 0.99, 0.99, 0.01, 0.99, 0.01], // input: [1, 0]
        [0.99, 0.99, 0.01, 0.99, 0.01, 0.01], // input: [1, 1]
    ];

    let mut nn = NeuralNetwork::new(vec![2, 4, 6]);  // 6 outputs: one for each logic gate

    let max_epochs = 5000;
    let mut error_threshold = 0.01;
    let learning_rate = 0.5;
    let decay_rate = 0.99;
    let final_filename = "nn_saved.json".to_string();

    let mut successfull = false;
    while !successfull {
        successfull = true;

        // Only need the outputs array now, no need to iterate over each logic gate
        println!("Training network for all logic gates simultaneously...");
        
        nn.problem = false; // Reset the issue flag
        while !nn.problem {
            nn.train(&inputs, &outputs, max_epochs, learning_rate, error_threshold, decay_rate);
            if !nn.problem {
                break;
            }
            println!("Training in progress... Neurons={:?}", nn.neuron_counts[1]);
        }

        // Evaluate the training result for all gates at once
        let final_error = nn.evaluate(&inputs, &outputs);
        println!("Final error after training all gates: {:.6}", final_error);

        if final_error <= error_threshold {
            println!("Training completed successfully!");
            successfull = true;
        } else {
            println!("Error still too high after training, retraining...");
            successfull = false;
        }

        // Save the final network
        if let Err(e) = nn.save_to_file(&final_filename) {
            eprintln!("Failed to save network: {}", e);
        } else {
            println!("Network saved to: {}", final_filename);
        }
    }
}
