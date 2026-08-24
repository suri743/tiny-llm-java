package com.surish.ai.llm.nn;

import com.surish.ai.llm.tensor.Vector;

public class TransformerBlock {

    private final int dim;
    public final SelfAttention selfAttention;
    public final LayerNorm norm1;
    public final FeedForwardLayer feedForward;
    public final LayerNorm norm2;

    // cached between forward and backward
    public Vector lastNorm1Output;
    public Vector lastFfnResidual;

    public TransformerBlock(int dim, int numHeads) {
        this.dim = dim;
        this.selfAttention = new SelfAttention(dim, numHeads);
        this.norm1 = new LayerNorm(dim);
        this.feedForward = new FeedForwardLayer(dim);
        this.norm2 = new LayerNorm(dim);
    }

    public Vector[] forward(Vector[] tokens) {
        // attention + residual + norm for all tokens
        Vector[] attended = selfAttention.forward(tokens);
        Vector[] norm1Out = new Vector[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            Vector residual = new Vector(dim);
            for (int d = 0; d < dim; d++)
                residual.set(d, attended[i].get(d) + tokens[i].get(d));
            norm1Out[i] = norm1.forward(residual);
        }

        // FFN + residual + norm for all tokens
        Vector[] output = new Vector[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            Vector ffnRaw = feedForward.forward(norm1Out[i]);
            Vector ffnResidual = new Vector(dim);
            for (int d = 0; d < dim; d++)
                ffnResidual.set(d, ffnRaw.get(d) + norm1Out[i].get(d));
            output[i] = norm2.forward(ffnResidual);
        }

        // cache last token's intermediates for backprop
        lastNorm1Output = norm1Out[tokens.length - 1];
        lastFfnResidual = output[tokens.length - 1];

        return output;
    }

    // returns gradient w.r.t block input (last token) for chaining blocks
    public Vector backward(Vector dOutput, double learningRate) {
        // grad through norm2
        Vector dFfnResidual = norm2.backward(dOutput, learningRate);
        // residual splits: ffn path + skip path
        Vector dFfnInput = feedForward.backward(dFfnResidual, learningRate);
        Vector dNorm1Out = add(dFfnInput, dFfnResidual);
        // grad through norm1
        Vector dAttResidual = norm1.backward(dNorm1Out, learningRate);
        // residual splits: attention path + skip path
        selfAttention.backward(dAttResidual, learningRate);
        return dAttResidual;
    }

    private Vector add(Vector a, Vector b) {
        Vector result = new Vector(a.size());
        for (int i = 0; i < a.size(); i++) result.set(i, a.get(i) + b.get(i));
        return result;
    }
}
