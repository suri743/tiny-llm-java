package com.surish.ai.llm.nn;

import com.surish.ai.llm.tensor.Vector;

public interface Layer {

    Vector forward(Vector input);
    Vector backward(Vector input, Vector outputGradient, double learningRate);

}