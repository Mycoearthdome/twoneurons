use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand::thread_rng;
use ndarray::concatenate;

#[derive(Debug)]
struct NeuralNetwork {
    weights: Vec<Array2<f64>>,  // List of weight matrices for each layer
    biases: Vec<Array1<f64>>,   // List of bias vectors for each layer
    neuron_counts: Vec<usize>,  // Neuron count for each layer
}

impl NeuralNetwork {
    fn new(neuron_counts: Vec<usize>) -> NeuralNetwork {
        let mut rng = thread_rng();
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
                Uniform::new(-0.1, 0.1), // Initialize biases to small random values
                &mut rng,
            );

            weights.push(weight);
            biases.push(bias);
        }

        NeuralNetwork {
            weights,
            biases,
            neuron_counts,
        }
    }

    fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut activation = input.clone(); // Start with the input as the first activation

        for (i, (weight, bias)) in self.weights.iter().zip(&self.biases).enumerate() {
            let weighted_input = weight.dot(&activation); // Matrix multiplication
            let weighted_input_with_bias = weighted_input + bias; // Add bias
            let new_activation = weighted_input_with_bias.map(|x| x.max(0.0)); // ReLU activation
            activation = new_activation;
        }

        activation
    }

    fn train(&mut self, inputs: &Array2<f64>, outputs: &Array2<f64>, epochs: usize, initial_learning_rate: f64, error_threshold: f64) {
        let mut learning_rate = initial_learning_rate;
        
        for epoch in 0..epochs {
            for (input, output) in inputs.rows().into_iter().zip(outputs.rows()) {
                let input = input.to_owned();
                let output = output.to_owned();
        
                // Forward pass
                let mut activation = input.clone();
                let mut activations = Vec::new();
                let mut zs = Vec::new();
        
                for (weight, bias) in self.weights.iter().zip(&self.biases) {
                    let z = weight.dot(&activation) + bias;
                    zs.push(z.clone());
                    let activation_val = if zs.len() < self.weights.len() {
                        z.map(|x| x.max(0.0))  // ReLU for hidden layers
                    } else {
                        let exp_vals = z.mapv(|x| x.exp()); // Apply exp to each element for Softmax
                        let sum_exp_vals = exp_vals.sum();
                        exp_vals / sum_exp_vals // Normalize to get probabilities
                    };
                    activations.push(activation_val.clone());
                    activation = activation_val;
                }
        
                // Backpropagation
                let mut deltas = Vec::new();
                let mut delta = output - activation; // Error at the output layer
                deltas.push(delta.clone()); // Clone to store it for use in the next layer
        
                // Backpropagate through each layer
                for i in (0..self.weights.len()).rev() {
                    let delta = deltas.last().unwrap();
                    let z = &zs[i];
                    let grad = delta * z.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }); // Derivative of ReLU
                    deltas.push(grad.clone()); // Clone to store it for use in the previous layer
    
                    let prev_activation = if i == 0 { input.clone() } else { activations[i - 1].clone() };
                    let weight_gradient = grad.clone().insert_axis(Axis(1)).dot(&prev_activation.insert_axis(Axis(0)));
                    self.weights[i] = &self.weights[i] - learning_rate * weight_gradient;
                    self.biases[i] = &self.biases[i] - learning_rate * grad.sum_axis(Axis(0));
                }
            }
    
            // After each epoch, check the error
            let error = self.evaluate(&inputs, &outputs);
            println!("Epoch: {}, Error: {} - Neurons: {:?}", epoch, error,self.neuron_counts[1]);
    
            // Stop adding neurons if the error is below the threshold
            if error < error_threshold {
                println!("Stopping early, error threshold met: {}", error_threshold);
                break;
            }

            // Optionally add a new neuron only if the error is still above threshold
            if error > error_threshold && epoch % 5000 == 0 {
                let last_hidden_layer_index = self.neuron_counts.len() - 3; // Skip input and output layers
                self.add_neuron_to_layer(last_hidden_layer_index);
                //println!("Added a new neuron. New network architecture: {:?}", self.neuron_counts);
            }

            // Decay the learning rate by a factor (e.g., 1.01 for slow decay)
            learning_rate /= 1.05;
        }
    }

    fn add_neuron_to_layer(&mut self, layer_index: usize) {
        assert!(layer_index < self.weights.len() - 1, "Cannot add neuron to output layer");

        // Update bias vector: add one more bias
        let new_bias = Array1::<f64>::zeros(1);  // A new bias value (initialized to zero)
        self.biases[layer_index] = concatenate![Axis(0), self.biases[layer_index].clone(), new_bias];

        // Update weight matrix for the layer receiving input (add a row to increase output neurons)
        let input_size = self.weights[layer_index].dim().1;
        let new_weights = Array1::random(input_size, Uniform::new(-0.5, 0.5));
        self.weights[layer_index].push_row(new_weights.view()).unwrap();

        // Update the *next* weight matrix: add a column for the new neuron
        let output_size = self.weights[layer_index + 1].dim().0;
        let new_out_weights = Array1::random(output_size, Uniform::new(-0.5, 0.5));
        self.weights[layer_index + 1].push_column(new_out_weights.view()).unwrap();

        // Update neuron count
        self.neuron_counts[layer_index + 1] += 1;
    }

    fn evaluate(&self, inputs: &Array2<f64>, outputs: &Array2<f64>) -> f64 {
        let predicted_outputs: Vec<Array1<f64>> = inputs.axis_iter(Axis(0))  // Iterate over rows
            .map(|input| self.forward(&input.to_owned())) // Clone the view to an owned array
            .collect();

        let error: f64 = predicted_outputs.iter().zip(outputs.rows()).map(|(predicted, actual)| {
            let diff = predicted.clone() - actual;  // Element-wise difference
            diff.mapv(|x| x.powi(2)).sum()  // Squared error for each example
        }).sum::<f64>() / (predicted_outputs.len() as f64 * predicted_outputs[0].len() as f64);

        error
    }
}

fn main() {
    // Example data (XOR problem, should add neurons dynamically for more complex tasks)
    let inputs = array![
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ];

    let outputs = array![
        [0.0],
        [1.0],
        [1.0],
        [0.0],
    ];

    let mut nn = NeuralNetwork::new(vec![2, 4, 1]); // Start with a simple 2-4-1 network

    let max_epochs = 1000000;
    let error_threshold = 0.01;
    let initial_learning_rate = 0.001;  // Start with a smaller learning rate

    nn.train(&inputs, &outputs, max_epochs, initial_learning_rate, error_threshold);  // Train for 10000 epochs


    // Final evaluation
    let error = nn.evaluate(&inputs, &outputs);
    println!("Final error: {}", error);
    println!("Final number of neurons necessary: {:?}", nn.neuron_counts[1])
}
