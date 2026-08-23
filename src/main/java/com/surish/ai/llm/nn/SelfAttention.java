package com.surish.ai.llm.nn;

import com.surish.ai.llm.tensor.Matrix;
import com.surish.ai.llm.tensor.Vector;

import java.util.Random;

public class SelfAttention {

    private final Matrix wQ;
    private final Matrix wK;
    private final Matrix wV;
    private final int dim;

    // cached from forward pass for backprop
    private Vector[] cachedTokens;
    private Vector[] cachedQ;
    private Vector[] cachedK;
    private Vector[] cachedV;
    private double[][] cachedScores;

    public SelfAttention(int dim) {
        this.dim = dim;
        this.wQ = randomMatrix(dim, dim);
        this.wK = randomMatrix(dim, dim);
        this.wV = randomMatrix(dim, dim);
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
            for (int j = 0; j < input.size(); j++) {
                sum += input.get(j) * W.get(j, i);
            }
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
        double scale = Math.sqrt(dim);

        Vector[] Q = new Vector[seqLen];
        Vector[] K = new Vector[seqLen];
        Vector[] V = new Vector[seqLen];
        for (int i = 0; i < seqLen; i++) {
            Q[i] = project(tokens[i], wQ);
            K[i] = project(tokens[i], wK);
            V[i] = project(tokens[i], wV);
        }

        double[][] scores = new double[seqLen][seqLen];
        Vector[] output = new Vector[seqLen];

        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < seqLen; j++) {
                scores[i][j] = dot(Q[i], K[j]) / scale;
            }

            // apply causal mask — block future tokens
            for (int j = i + 1; j < seqLen; j++) {
                scores[i][j] = Double.NEGATIVE_INFINITY;
            }

            double max = scores[i][0];
            for (double s : scores[i]) if (s > max) max = s;
            double sum = 0.0;
            for (int j = 0; j < seqLen; j++) {
                scores[i][j] = Math.exp(scores[i][j] - max);
                sum += scores[i][j];
            }
            for (int j = 0; j < seqLen; j++) scores[i][j] /= sum;

            Vector out = new Vector(dim);
            for (int j = 0; j < seqLen; j++) {
                for (int d = 0; d < dim; d++) {
                    out.set(d, out.get(d) + scores[i][j] * V[j].get(d));
                }
            }
            output[i] = out;
        }

        // cache for backprop
        cachedTokens = tokens;
        cachedQ = Q;
        cachedK = K;
        cachedV = V;
        cachedScores = scores;

        return output;
    }

    // dOutput: gradient w.r.t. output of last token (Vector[dim])
    public void backward(Vector dOutput, double learningRate) {
        int seqLen = cachedTokens.length;
        double scale = Math.sqrt(dim);

        // we only backprop through the last token's output
        int i = seqLen - 1;

        // gradient w.r.t V: dV[j] = scores[i][j] * dOutput
        Vector[] dV = new Vector[seqLen];
        for (int j = 0; j < seqLen; j++) {
            dV[j] = new Vector(dim);
            for (int d = 0; d < dim; d++) {
                dV[j].set(d, cachedScores[i][j] * dOutput.get(d));
            }
        }

        // gradient w.r.t scores[i][j]: dScores[j] = dOutput · V[j]
        double[] dScores = new double[seqLen];
        for (int j = 0; j < seqLen; j++) {
            dScores[j] = dot(dOutput, cachedV[j]);
        }

        // backprop through softmax: dRawScores[j] = scores[j] * (dScores[j] - sum(scores * dScores))
        double dotSD = 0.0;
        for (int j = 0; j < seqLen; j++) dotSD += cachedScores[i][j] * dScores[j];
        double[] dRawScores = new double[seqLen];
        for (int j = 0; j < seqLen; j++) {
            dRawScores[j] = cachedScores[i][j] * (dScores[j] - dotSD) / scale;
        }

        // gradient w.r.t Q[i]: dQ = sum_j(dRawScores[j] * K[j])
        Vector dQ = new Vector(dim);
        for (int j = 0; j < seqLen; j++) {
            for (int d = 0; d < dim; d++) {
                dQ.set(d, dQ.get(d) + dRawScores[j] * cachedK[j].get(d));
            }
        }

        // gradient w.r.t K[j]: dK[j] = dRawScores[j] * Q[i]
        Vector[] dK = new Vector[seqLen];
        for (int j = 0; j < seqLen; j++) {
            dK[j] = new Vector(dim);
            for (int d = 0; d < dim; d++) {
                dK[j].set(d, dRawScores[j] * cachedQ[i].get(d));
            }
        }

        // update wV: for each token j, wV += -lr * outer(tokens[j], dV[j])
        for (int j = 0; j < seqLen; j++) {
            for (int row = 0; row < dim; row++) {
                for (int col = 0; col < dim; col++) {
                    wV.set(row, col, wV.get(row, col) - learningRate * cachedTokens[j].get(row) * dV[j].get(col));
                }
            }
        }

        // update wQ: wQ += -lr * outer(tokens[i], dQ)
        for (int row = 0; row < dim; row++) {
            for (int col = 0; col < dim; col++) {
                wQ.set(row, col, wQ.get(row, col) - learningRate * cachedTokens[i].get(row) * dQ.get(col));
            }
        }

        // update wK: for each token j, wK += -lr * outer(tokens[j], dK[j])
        for (int j = 0; j < seqLen; j++) {
            for (int row = 0; row < dim; row++) {
                for (int col = 0; col < dim; col++) {
                    wK.set(row, col, wK.get(row, col) - learningRate * cachedTokens[j].get(row) * dK[j].get(col));
                }
            }
        }
    }
}
