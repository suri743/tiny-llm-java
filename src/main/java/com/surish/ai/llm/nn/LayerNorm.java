package com.surish.ai.llm.nn;

import com.surish.ai.llm.tensor.Vector;

public class LayerNorm {

    private final int dim;
    private final double[] gamma;
    private final double[] beta;

    // cached last-token state for backprop
    private Vector cachedInput;
    private double cachedMean;
    private double cachedStd;
    private Vector cachedNormalized;

    public LayerNorm(int dim) {
        this.dim = dim;
        this.gamma = new double[dim];
        this.beta = new double[dim];
        for (int i = 0; i < dim; i++) {
            gamma[i] = 1.0;
            beta[i] = 0.0;
        }
    }

    public Vector forward(Vector input) {
        double mean = 0.0;
        for (int i = 0; i < dim; i++) mean += input.get(i);
        mean /= dim;

        double variance = 0.0;
        for (int i = 0; i < dim; i++) {
            double diff = input.get(i) - mean;
            variance += diff * diff;
        }
        variance /= dim;
        double std = Math.sqrt(variance + 1e-5);

        Vector normalized = new Vector(dim);
        for (int i = 0; i < dim; i++)
            normalized.set(i, (input.get(i) - mean) / std);

        Vector output = new Vector(dim);
        for (int i = 0; i < dim; i++)
            output.set(i, gamma[i] * normalized.get(i) + beta[i]);

        // cache for backprop (caller must call forwardAndCache on the last token)
        cachedInput = input;
        cachedMean = mean;
        cachedStd = std;
        cachedNormalized = normalized;

        return output;
    }

    public Vector backward(Vector dOutput, double learningRate) {
        for (int i = 0; i < dim; i++) {
            gamma[i] -= learningRate * dOutput.get(i) * cachedNormalized.get(i);
            beta[i]  -= learningRate * dOutput.get(i);
        }

        Vector dNorm = new Vector(dim);
        for (int i = 0; i < dim; i++)
            dNorm.set(i, dOutput.get(i) * gamma[i]);

        double dVar = 0.0;
        for (int i = 0; i < dim; i++)
            dVar += dNorm.get(i) * (cachedInput.get(i) - cachedMean);
        dVar *= -0.5 / (cachedStd * cachedStd * cachedStd);

        double dMean = 0.0;
        for (int i = 0; i < dim; i++)
            dMean += dNorm.get(i) * (-1.0 / cachedStd) + dVar * (-2.0 * (cachedInput.get(i) - cachedMean) / dim);

        Vector dInput = new Vector(dim);
        for (int i = 0; i < dim; i++) {
            dInput.set(i,
                dNorm.get(i) / cachedStd
                + dVar * 2.0 * (cachedInput.get(i) - cachedMean) / dim
                + dMean / dim
            );
        }

        return dInput;
    }
}
