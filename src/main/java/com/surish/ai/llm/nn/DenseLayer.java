package com.surish.ai.llm.nn;

import com.surish.ai.llm.tensor.Vector;

import java.util.ArrayList;
import java.util.List;

public class DenseLayer implements Layer {

    private final List<Neuron> neurons = new ArrayList<>();

    public DenseLayer(int inputSize, int outputSize) {

        for (int i = 0; i < outputSize; i++) {
            neurons.add(new LinearNeuron(inputSize));
        }
    }

    @Override
    public Vector forward(Vector input) {

        Vector output = new Vector(neurons.size());

        for (int i = 0; i < neurons.size(); i++) {
            output.set(i, neurons.get(i).forward(input));
        }

        return output;
    }

    @Override
    public Vector backward(Vector input, Vector outputGradient, double learningRate) {
        Vector inputGradient = new Vector(input.size());
        for (int i = 0; i < neurons.size(); i++) {
            Vector neuronGradient = neurons.get(i).backward(input, outputGradient.get(i), learningRate);
            for (int j = 0; j < input.size(); j++) {
                inputGradient.set(j, inputGradient.get(j) + neuronGradient.get(j));
            }
        }
        return inputGradient;
    }
}