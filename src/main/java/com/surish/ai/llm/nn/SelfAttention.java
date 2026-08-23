package com.surish.ai.llm.nn;

import com.surish.ai.llm.tensor.Matrix;
import com.surish.ai.llm.tensor.Vector;

import java.util.Random;

public class SelfAttention {

    private final int dim;
    private final int numHeads;
    private final int headDim;

    // one W_Q, W_K, W_V per head
    private final Matrix[] wQ;
    private final Matrix[] wK;
    private final Matrix[] wV;

    // output projection: concatenated heads → dim
    private final Matrix wO;

    // cached for backprop
    private Vector[] cachedTokens;
    private Vector[][] cachedQ;   // [head][token]
    private Vector[][] cachedK;
    private Vector[][] cachedV;
    private double[][][] cachedScores; // [head][token_i][token_j]
    private Vector[][] cachedHeadOut;  // [head][token]

    public SelfAttention(int dim, int numHeads) {
        this.dim = dim;
        this.numHeads = numHeads;
        this.headDim = dim / numHeads;

        wQ = new Matrix[numHeads];
        wK = new Matrix[numHeads];
        wV = new Matrix[numHeads];
        for (int h = 0; h < numHeads; h++) {
            wQ[h] = randomMatrix(dim, headDim);
            wK[h] = randomMatrix(dim, headDim);
            wV[h] = randomMatrix(dim, headDim);
        }
        wO = randomMatrix(dim, dim);
    }

    private Matrix randomMatrix(int rows, int cols) {
        Matrix m = new Matrix(rows, cols);
        Random random = new Random(42);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m.set(i, j, random.nextGaussian() * 0.02);
        return m;
    }

    private Vector project(Vector input, Matrix W) {
        Vector output = new Vector(W.columns());
        for (int i = 0; i < W.columns(); i++) {
            double sum = 0.0;
            for (int j = 0; j < input.size(); j++) sum += input.get(j) * W.get(j, i);
            output.set(i, sum);
        }
        return output;
    }

    private double dot(Vector a, Vector b) {
        double sum = 0.0;
        for (int i = 0; i < a.size(); i++) sum += a.get(i) * b.get(i);
        return sum;
    }

    public Vector[] forward(Vector[] tokens) {
        int seqLen = tokens.length;
        double scale = Math.sqrt(headDim);

        cachedQ = new Vector[numHeads][seqLen];
        cachedK = new Vector[numHeads][seqLen];
        cachedV = new Vector[numHeads][seqLen];
        cachedScores = new double[numHeads][seqLen][seqLen];
        cachedHeadOut = new Vector[numHeads][seqLen];
        cachedTokens = tokens;

        // run each head independently
        for (int h = 0; h < numHeads; h++) {
            for (int i = 0; i < seqLen; i++) {
                cachedQ[h][i] = project(tokens[i], wQ[h]);
                cachedK[h][i] = project(tokens[i], wK[h]);
                cachedV[h][i] = project(tokens[i], wV[h]);
            }

            for (int i = 0; i < seqLen; i++) {
                for (int j = 0; j < seqLen; j++) {
                    cachedScores[h][i][j] = dot(cachedQ[h][i], cachedK[h][j]) / scale;
                }

                // causal mask
                for (int j = i + 1; j < seqLen; j++) {
                    cachedScores[h][i][j] = Double.NEGATIVE_INFINITY;
                }

                // softmax
                double max = cachedScores[h][i][0];
                for (double s : cachedScores[h][i]) if (s > max) max = s;
                double sum = 0.0;
                for (int j = 0; j < seqLen; j++) {
                    cachedScores[h][i][j] = Math.exp(cachedScores[h][i][j] - max);
                    sum += cachedScores[h][i][j];
                }
                for (int j = 0; j < seqLen; j++) cachedScores[h][i][j] /= sum;

                // weighted sum of V
                Vector out = new Vector(headDim);
                for (int j = 0; j < seqLen; j++) {
                    for (int d = 0; d < headDim; d++) {
                        out.set(d, out.get(d) + cachedScores[h][i][j] * cachedV[h][j].get(d));
                    }
                }
                cachedHeadOut[h][i] = out;
            }
        }

        // concatenate heads and project through wO
        Vector[] output = new Vector[seqLen];
        for (int i = 0; i < seqLen; i++) {
            Vector concat = new Vector(dim);
            for (int h = 0; h < numHeads; h++) {
                for (int d = 0; d < headDim; d++) {
                    concat.set(h * headDim + d, cachedHeadOut[h][i].get(d));
                }
            }
            output[i] = project(concat, wO);
        }

        return output;
    }

    public void backward(Vector dOutput, double learningRate) {
        int seqLen = cachedTokens.length;
        double scale = Math.sqrt(headDim);

        // backprop through wO for last token
        int lastToken = seqLen - 1;

        // reconstruct concat for last token
        Vector concat = new Vector(dim);
        for (int h = 0; h < numHeads; h++) {
            for (int d = 0; d < headDim; d++) {
                concat.set(h * headDim + d, cachedHeadOut[h][lastToken].get(d));
            }
        }

        // gradient w.r.t concat = wO * dOutput  (wO is dim×dim, dOutput is dim)
        Vector dConcat = new Vector(dim);
        for (int row = 0; row < dim; row++) {
            double sum = 0.0;
            for (int col = 0; col < dim; col++) {
                sum += wO.get(row, col) * dOutput.get(col);
            }
            dConcat.set(row, sum);
        }

        // update wO
        for (int row = 0; row < dim; row++) {
            for (int col = 0; col < dim; col++) {
                wO.set(row, col, wO.get(row, col) - learningRate * concat.get(row) * dOutput.get(col));
            }
        }

        // backprop through each head
        for (int h = 0; h < numHeads; h++) {
            // gradient for this head's output at last token
            Vector dHeadOut = new Vector(headDim);
            for (int d = 0; d < headDim; d++) {
                dHeadOut.set(d, dConcat.get(h * headDim + d));
            }

            // backprop through all tokens for this head
            for (int i = 0; i < seqLen; i++) {
                Vector dV_i = new Vector(headDim);
                for (int j = 0; j <= i; j++) {
                    for (int d = 0; d < headDim; d++) {
                        dV_i.set(d, dV_i.get(d) + cachedScores[h][i][j] * dHeadOut.get(d));
                    }
                }

                double[] dScores = new double[seqLen];
                for (int j = 0; j <= i; j++) {
                    dScores[j] = dot(dHeadOut, cachedV[h][j]);
                }

                double dotSD = 0.0;
                for (int j = 0; j <= i; j++) dotSD += cachedScores[h][i][j] * dScores[j];
                double[] dRaw = new double[seqLen];
                for (int j = 0; j <= i; j++) {
                    dRaw[j] = cachedScores[h][i][j] * (dScores[j] - dotSD) / scale;
                }

                Vector dQ = new Vector(headDim);
                for (int j = 0; j <= i; j++) {
                    for (int d = 0; d < headDim; d++) {
                        dQ.set(d, dQ.get(d) + dRaw[j] * cachedK[h][j].get(d));
                    }
                }

                // update wQ, wK, wV for this head
                for (int row = 0; row < dim; row++) {
                    for (int d = 0; d < headDim; d++) {
                        wQ[h].set(row, d, wQ[h].get(row, d) - learningRate * cachedTokens[i].get(row) * dQ.get(d));
                        wV[h].set(row, d, wV[h].get(row, d) - learningRate * cachedTokens[i].get(row) * dV_i.get(d));
                    }
                }

                for (int j = 0; j <= i; j++) {
                    for (int row = 0; row < dim; row++) {
                        for (int d = 0; d < headDim; d++) {
                            wK[h].set(row, d, wK[h].get(row, d) - learningRate * cachedTokens[j].get(row) * dRaw[j] * cachedQ[h][i].get(d));
                        }
                    }
                }
            }
        }
    }
}
