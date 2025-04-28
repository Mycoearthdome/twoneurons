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

        for i in 0..neuron_counts.len() - 1 {
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
        for (i, (weight, bias)) in self.weights.iter().zip(&self.biases).enumerate() {
            let z = weight.dot(&activation) + bias;
            if i == self.weights.len() - 1 {
                // Output layer (no softmax, using MSE loss)
                activation = z;
            } else {
                activation = Self::relu(&z);
            }
        }
        activation
    }

    fn train(
        &mut self,
        inputs: &Array2<f64>,
        outputs: &Array2<f64>,
        epochs: usize,
        mut learning_rate: f64,
        error_threshold: f64,
        logic_gate_name: &str,
    ) {
        self.problem = false;
        for _epoch in 0..epochs {
            for (input, output) in inputs.outer_iter().zip(outputs.outer_iter()) {
                let input = input.to_owned();
                let output = output.to_owned();

                let mut activations = vec![input.clone()];
                let mut zs = Vec::new();

                // Forward pass
                let mut activation = input.clone();
                for (i, (weight, bias)) in self.weights.iter().zip(&self.biases).enumerate() {
                    let z = weight.dot(&activation) + bias;
                    zs.push(z.clone());
                    activation = if i == self.weights.len() - 1 {
                        z
                    } else {
                        Self::relu(&z)
                    };
                    activations.push(activation.clone());
                }

                // Backward pass
                let mut delta = activations.last().unwrap() - &output;
                for i in (0..self.weights.len()).rev() {
                    let z = &zs[i];
                    if i != self.weights.len() - 1 {
                        delta = delta * Self::relu_derivative(&z);
                    }

                    let prev_activation = &activations[i];
                    let weight_grad = delta.view().insert_axis(Axis(1))
                        .dot(&prev_activation.view().insert_axis(Axis(0)));

                    self.weights[i] = self.weights[i].clone() - (learning_rate * weight_grad);
                    self.biases[i] = self.biases[i].clone() - (learning_rate * delta.clone());
                }
            }

            let error = self.evaluate(inputs, outputs);
            if error <= error_threshold {
                println!("Training succeeded for {} gate, error threshold met.", logic_gate_name);
                break;
            }

            if error > 1.0 {
                self.problem = true;
                continue;
            }

            // Add neuron dynamically if still not below threshold
            if error > error_threshold {
                let last_hidden_layer = self.neuron_counts.len() - 3;
                self.add_neuron_to_layer(last_hidden_layer);
            }

            learning_rate /= 1.05; // learning rate decay
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
        inputs
            .outer_iter()
            .zip(outputs.outer_iter())
            .map(|(input, output)| {
                let prediction = self.forward(&input.to_owned());
                (&prediction - &output).mapv(|x| x.powi(2)).sum()
            })
            .sum::<f64>()
            / (inputs.nrows() as f64 * outputs.ncols() as f64)
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
}

fn main() {
    let inputs = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];

    let logic_gates = vec![
        (array![[0.0], [0.0], [0.0], [1.0]], "AND"),
        (array![[0.0], [1.0], [1.0], [1.0]], "OR"),
        (array![[1.0], [1.0], [1.0], [0.0]], "NAND"),
        (array![[1.0], [0.0], [0.0], [0.0]], "NOR"),
        (array![[0.0], [1.0], [1.0], [0.0]], "XOR"),
        (array![[1.0], [0.0], [0.0], [1.0]], "XNOR"),
    ];

    let mut nn = NeuralNetwork::new(vec![2, 4, 1]);

    let max_epochs = 10_000;
    let error_threshold = 0.01;
    let learning_rate = 0.001;
    let mut filename = <String>::new();
    let final_filename ="nn_saved.json".to_string();

    for (outputs, gate_name) in logic_gates {
        let oldgate_filename = filename.clone();
        let mut path = Path::new(&final_filename);
        if path.exists(){
            match NeuralNetwork::load_from_file(&final_filename) {
                Ok(nn) => {
                    println!("Successfully loaded the network.");
                    
                    let final_error = nn.evaluate(&inputs, &outputs);
                    println!("Final error after learning {} gate: {:.6}", gate_name, final_error);                    

                }
                Err(e) => {
                        eprintln!("Failed to load network: {}", e);
                }
            }
        } else {
            println!("Training network to learn {} gate:", gate_name);
            while !nn.problem {
                nn.train(&inputs, &outputs, max_epochs, learning_rate, error_threshold, gate_name);
                if !nn.problem {
                    break;
                }
                println!("Retrying training with simpler configuration for {} gate...", gate_name);
                path = Path::new(&oldgate_filename);
                if path.exists(){
                    match NeuralNetwork::load_from_file(&oldgate_filename) {
                        Ok(nn_progress) => {
                            //println!("Successfully loaded the network's last step.");
                            nn = nn_progress;
                            nn.problem = false; //reinit.
                            continue                  
                        }
                        Err(e) => {
                                eprintln!("Failed to load network: {}", e);
                        }
                    }
                } else {
                    nn = NeuralNetwork::new(vec![2, 4, 1]); // Retry
                }
            }

            // Save the neural network after training
            filename = format!("nn_{}.json", gate_name);

            if let Err(e) = nn.save_to_file(&filename) {
                eprintln!("Failed to save network: {}", e);
            } else {
                println!("Network saved to: {}", filename);
            }


        }
       
    }

    

    // final save of the neural network progress...
    let filename_final = format!("nn_saved.json");

    if let Err(e) = nn.save_to_file(&filename_final) {
        eprintln!("Failed to save network: {}", e);
    } else {
        println!("Network saved to: {}", filename_final);
    }
}
