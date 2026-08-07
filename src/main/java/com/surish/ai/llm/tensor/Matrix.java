package com.surish.ai.llm.tensor;

import lombok.EqualsAndHashCode;
import lombok.Getter;

@Getter
@EqualsAndHashCode
public class Matrix {

    private final double[][] values;

    public Matrix(int rows, int columns) {
        this.values = new double[rows][columns];
    }

    public Matrix(double[][] values) {

        this.values = new double[values.length][];

        for (int i = 0; i < values.length; i++) {
            this.values[i] = values[i].clone();
        }
    }

    public int rows() {
        return values.length;
    }

    public int columns() {
        return values[0].length;
    }

    public double get(int row, int column) {
        return values[row][column];
    }

    public void set(int row, int column, double value) {
        values[row][column] = value;
    }

    public Vector row(int index) {
        return new Vector(values[index]);
    }
}