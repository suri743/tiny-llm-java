package com.surish.ai.llm.nn;

import com.surish.ai.llm.tensor.Vector;

import java.util.Random;

public class LinearNeuron implements Neuron {

    private final Vector weights;

    private final double bias;

    public LinearNeuron(int inputSize) {

        Random random = new Random(42);

        weights = new Vector(inputSize);

        for (int i = 0; i < inputSize; i++) {
            weights.set(i, random.nextGaussian() * 0.02);
        }

        bias = random.nextGaussian() * 0.02;
    }

    @Override
    public double forward(Vector input) {

        if (input.size() != weights.size()) {
            throw new IllegalArgumentException("Input size mismatch.");
        }

        double sum = bias;

        for (int i = 0; i < input.size(); i++) {
            sum += input.get(i) * weights.get(i);
        }

        return sum;
    }
}