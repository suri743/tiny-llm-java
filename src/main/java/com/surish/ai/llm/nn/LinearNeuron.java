package com.surish.ai.llm.nn;

import com.surish.ai.llm.tensor.Vector;

import java.util.Random;

public class LinearNeuron implements Neuron {

    private static final Random random = new Random();

    private final Vector weights;
    private double bias;

    public LinearNeuron(int inputSize) {

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

    @Override
    public Vector backward(Vector input, double outputGradient,
                         double learningRate) {

        for (int i = 0; i < weights.size(); i++) {

            double weightGradient =
                outputGradient * input.get(i);

            double newWeight =
                weights.get(i)
                - learningRate * weightGradient;

            weights.set(i, newWeight);
        }

        bias -= learningRate * outputGradient;

        Vector inputGradient = new Vector(weights.size());
        for (int i = 0; i < weights.size(); i++) {
            inputGradient.set(i, outputGradient * weights.get(i));
        }
        return inputGradient;
    }
}