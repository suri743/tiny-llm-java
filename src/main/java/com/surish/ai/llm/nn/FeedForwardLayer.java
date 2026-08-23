package com.surish.ai.llm.nn;

import com.surish.ai.llm.tensor.Matrix;
import com.surish.ai.llm.tensor.Vector;

import java.util.Random;

public class FeedForwardLayer {

    private final int inputDim;
    private final int hiddenDim;

    private final Matrix w1; // inputDim × hiddenDim
    private final Matrix w2; // hiddenDim × inputDim

    private Vector cachedInput;
    private Vector cachedHidden; // after ReLU

    public FeedForwardLayer(int inputDim) {
        this.inputDim = inputDim;
        this.hiddenDim = inputDim * 4;
        this.w1 = randomMatrix(inputDim, hiddenDim);
        this.w2 = randomMatrix(hiddenDim, inputDim);
    }

    private Matrix randomMatrix(int rows, int cols) {
        Matrix m = new Matrix(rows, cols);
        Random random = new Random();
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m.set(i, j, random.nextGaussian() * 0.02);
        return m;
    }

    public Vector forward(Vector input) {
        cachedInput = input;

        // hidden = ReLU(input @ W1)
        Vector hidden = new Vector(hiddenDim);
        for (int j = 0; j < hiddenDim; j++) {
            double sum = 0.0;
            for (int i = 0; i < inputDim; i++) sum += input.get(i) * w1.get(i, j);
            hidden.set(j, Math.max(0.0, sum)); // ReLU
        }
        cachedHidden = hidden;

        // output = hidden @ W2
        Vector output = new Vector(inputDim);
        for (int i = 0; i < inputDim; i++) {
            double sum = 0.0;
            for (int j = 0; j < hiddenDim; j++) sum += hidden.get(j) * w2.get(j, i);
            output.set(i, sum);
        }
        return output;
    }

    // returns gradient w.r.t. input for further backprop
    public Vector backward(Vector dOutput, double learningRate) {
        // gradient w.r.t w2: outer(cachedHidden, dOutput)
        // gradient w.r.t hidden: dOutput @ W2^T
        Vector dHidden = new Vector(hiddenDim);
        for (int j = 0; j < hiddenDim; j++) {
            double sum = 0.0;
            for (int i = 0; i < inputDim; i++) sum += w2.get(j, i) * dOutput.get(i);
            dHidden.set(j, sum);
        }

        // update w2
        for (int j = 0; j < hiddenDim; j++)
            for (int i = 0; i < inputDim; i++)
                w2.set(j, i, w2.get(j, i) - learningRate * cachedHidden.get(j) * dOutput.get(i));

        // backprop through ReLU: zero gradient where hidden was <= 0 (before ReLU)
        // cachedHidden already has ReLU applied, so where hidden=0 → gate closed
        for (int j = 0; j < hiddenDim; j++)
            if (cachedHidden.get(j) <= 0.0) dHidden.set(j, 0.0);

        // gradient w.r.t input: dHidden @ W1^T
        Vector dInput = new Vector(inputDim);
        for (int i = 0; i < inputDim; i++) {
            double sum = 0.0;
            for (int j = 0; j < hiddenDim; j++) sum += w1.get(i, j) * dHidden.get(j);
            dInput.set(i, sum);
        }

        // update w1
        for (int i = 0; i < inputDim; i++)
            for (int j = 0; j < hiddenDim; j++)
                w1.set(i, j, w1.get(i, j) - learningRate * cachedInput.get(i) * dHidden.get(j));

        return dInput;
    }
}
