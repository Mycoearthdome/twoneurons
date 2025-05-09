use ndarray::concatenate;
use ndarray::prelude::*;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{self, Write, Read};
use ndarray_rand::rand_distr::{Normal, Distribution};
use std::process::exit;
use rayon::prelude::*;
use ndarray::{Array1, Array2, Axis, stack};

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

fn generate_realistic_drone_inputs(num_samples: usize) -> Array2<f64> {
    let mut rng = StdRng::from_entropy();

    let pos_range = Uniform::new(-1000.0, 1000.0);       // meters
    let angle_range = Uniform::new(-180.0, 180.0);       // degrees
    let vel_range = Uniform::new(-30.0, 30.0);           // m/s
    let acc_range = Uniform::new(-15.0, 15.0);           // m/s²
    let ang_vel_range = Uniform::new(-300.0, 300.0);     // deg/s
    let battery_range = Uniform::new(0.0, 1.0);          // normalized
    let control_range = Uniform::new(0.0, 1.0);          // throttle, etc.

    Array2::from_shape_fn((num_samples, 18), |(_, j)| match j {
        0..=2 => pos_range.sample(&mut rng),      // x, y, z
        3..=5 => angle_range.sample(&mut rng),    // roll, pitch, yaw
        6..=8 => vel_range.sample(&mut rng),      // vx, vy, vz
        9..=11 => acc_range.sample(&mut rng),     // ax, ay, az
        12..=14 => ang_vel_range.sample(&mut rng),// ωx, ωy, ωz
        15 => battery_range.sample(&mut rng),     // battery level
        16..=17 => control_range.sample(&mut rng),// control inputs (e.g. throttle, pitch)
        _ => 0.0,
    })
}

fn generate_realistic_outputs(inputs: &Array2<f64>) -> Array2<f64> {
    let rows: Vec<Array1<f64>> = inputs.outer_iter().map(|input| {
        let z = input[2];
        let vz = input[8];
        let vx = input[6];
        let vy = input[7];
        let ax = input[9];
        let ay = input[10];
        let yaw = input[5];
        let omega_z = input[14];

        let mut throttle = 0.5 + (-z + vz) * 0.01;
        let mut pitch = 0.5 + (-vx + ax) * 0.01;
        let mut roll = 0.5 + (-vy + ay) * 0.01;
        let mut yaw_rate = 0.5 + (-yaw + omega_z) * 0.005;

        for val in [&mut throttle, &mut pitch, &mut roll, &mut yaw_rate] {
            *val = val.clamp(0.0, 1.0);
        }

        array![throttle, pitch, roll, yaw_rate]
    }).collect();

    stack(Axis(0), &rows.iter().map(|x| x.view()).collect::<Vec<_>>()).unwrap()
}



fn get_input_feature_ranges() -> Vec<(f64, f64)> {
    vec![
        (-1000.0, 1000.0), // x
        (-1000.0, 1000.0), // y
        (-1000.0, 1000.0), // z
        (-180.0, 180.0),   // roll
        (-180.0, 180.0),   // pitch
        (-180.0, 180.0),   // yaw
        (-30.0, 30.0),     // vx
        (-30.0, 30.0),     // vy
        (-30.0, 30.0),     // vz
        (-15.0, 15.0),     // ax
        (-15.0, 15.0),     // ay
        (-15.0, 15.0),     // az
        (-300.0, 300.0),   // ωx
        (-300.0, 300.0),   // ωy
        (-300.0, 300.0),   // ωz
        (0.0, 1.0),        // battery
        (0.0, 1.0),        // throttle hint
        (0.0, 1.0),        // pitch hint
    ]
}

fn normalize_inputs(inputs: &Array2<f64>) -> Array2<f64> {
    let ranges = get_input_feature_ranges();
    let (rows, cols) = inputs.dim();
    assert_eq!(cols, ranges.len());

    Array2::from_shape_fn((rows, cols), |(i, j)| {
        let (min, max) = ranges[j];
        let val = inputs[[i, j]];
        ((val - min) / (max - min)).clamp(0.0, 1.0)
    })
}

fn interpret_control(name: &str, value: f64) -> &'static str {
    match name {
        "throttle" => {
            if value < 0.45 {
                "reduce lift (descending)"
            } else if value > 0.55 {
                "increase lift (ascending)"
            } else {
                "maintain altitude (hover)"
            }
        },
        "pitch" => {
            if value < 0.45 {
                "tilt backward"
            } else if value > 0.55 {
                "tilt forward"
            } else {
                "stay level"
            }
        },
        "roll" => {
            if value < 0.45 {
                "tilt left"
            } else if value > 0.55 {
                "tilt right"
            } else {
                "stay level"
            }
        },
        "yaw" => {
            if value < 0.45 {
                "rotate counterclockwise"
            } else if value > 0.55 {
                "rotate clockwise"
            } else {
                "no turn"
            }
        },
        _ => "unknown"
    }
}

fn main() {
    let num_samples = 1000;
    let input_dim = 18;
    let output_dim = 4; // Example: motor command outputs (throttle, yaw, pitch, roll)

    // Simulate realistic drone inputs
    let raw_inputs = generate_realistic_drone_inputs(num_samples);
    let inputs = normalize_inputs(&raw_inputs); // normalized to [0.0, 1.0]

    // Simulate ideal motor command outputs for learning
    let outputs = generate_realistic_outputs(&inputs);


    let mut nn = NeuralNetwork::new(vec![input_dim, 12, 8, output_dim]);

    let max_epochs = 10_000;
    let learning_rate = 0.01;
    let error_threshold = 0.001;
    let decay_rate = 0.99;
    let final_filename = "realistic_drone_nn.json";

    println!("Training neural network with realistic drone flight data...");
    nn.train(&inputs, &outputs, max_epochs, learning_rate, error_threshold, decay_rate);

    let final_error = nn.evaluate(&inputs, &outputs);
    // Convert final error to RMSE (Root Mean Squared Error)
    let rmse = final_error.sqrt();
    
    // Convert RMSE to percentage (assuming outputs are normalized between 0 and 1)
    let rmse_percentage = rmse * 100.0;

    println!("Final training error (RMSE in percentage): {:.2}%", rmse_percentage);

    if let Err(e) = nn.save_to_file(final_filename) {
        eprintln!("Failed to save network: {}", e);
    } else {
        println!("Network saved to: {}", final_filename);
    }

    // example Prediction
    let sample_input = inputs.row(0).to_owned();
    let result = nn.forward(&sample_input);
    println!("\n--- Sample Prediction ---");
    //println!("Sample normalized input (telemetry): {:?}", sample_input);
    println!("Predicted control output (normalized values in [0.0 - 1.0]):");
    println!("  Throttle : {:.4} → {}",
        result[0],
        interpret_control("throttle", result[0])
    );
    println!("  Pitch    : {:.4} → {}",
        result[1],
        interpret_control("pitch", result[1])
    );
    println!("  Roll     : {:.4} → {}",
        result[2],
        interpret_control("roll", result[2])
    );
    println!("  Yaw Rate : {:.4} → {}",
        result[3],
        interpret_control("yaw", result[3])
    );
}
