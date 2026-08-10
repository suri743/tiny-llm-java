package com.surish.ai.llm.nn;

import com.surish.ai.llm.tensor.Vector;

public interface Neuron {

    double forward(Vector input);

    void backward(Vector input, double outputGradient, double learningRate);

}