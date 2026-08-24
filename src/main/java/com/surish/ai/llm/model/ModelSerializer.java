package com.surish.ai.llm.model;

import com.surish.ai.llm.nn.DenseLayer;
import com.surish.ai.llm.nn.FeedForwardLayer;
import com.surish.ai.llm.nn.LayerNorm;
import com.surish.ai.llm.nn.LinearNeuron;
import com.surish.ai.llm.nn.SelfAttention;
import com.surish.ai.llm.nn.TransformerBlock;
import com.surish.ai.llm.tensor.Matrix;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.lang.reflect.Field;
import java.util.List;

public class ModelSerializer {

    public static void save(LanguageModel model, String path) throws IOException {
        try (DataOutputStream out = new DataOutputStream(new FileOutputStream(path))) {
            writeMatrix(out, getMatrix(model.embeddingLayer, "embeddings"));
            writeMatrix(out, getMatrix(model.positionalEmbeddingLayer, "embeddings"));
            for (TransformerBlock block : model.blocks) {
                writeSelfAttention(out, block.selfAttention);
                writeLayerNorm(out, block.norm1);
                writeFeedForward(out, block.feedForward);
                writeLayerNorm(out, block.norm2);
            }
            writeDenseLayer(out, model.outputLayer);
        }
    }

    public static void load(LanguageModel model, String path) throws IOException {
        try (DataInputStream in = new DataInputStream(new FileInputStream(path))) {
            readMatrix(in, getMatrix(model.embeddingLayer, "embeddings"));
            readMatrix(in, getMatrix(model.positionalEmbeddingLayer, "embeddings"));
            for (TransformerBlock block : model.blocks) {
                readSelfAttention(in, block.selfAttention);
                readLayerNorm(in, block.norm1);
                readFeedForward(in, block.feedForward);
                readLayerNorm(in, block.norm2);
            }
            readDenseLayer(in, model.outputLayer);
        }
    }

    private static void writeSelfAttention(DataOutputStream out, SelfAttention sa) throws IOException {
        Matrix[] wQ = getField(sa, "wQ");
        Matrix[] wK = getField(sa, "wK");
        Matrix[] wV = getField(sa, "wV");
        Matrix   wO = getField(sa, "wO");
        for (Matrix m : wQ) writeMatrix(out, m);
        for (Matrix m : wK) writeMatrix(out, m);
        for (Matrix m : wV) writeMatrix(out, m);
        writeMatrix(out, wO);
    }

    private static void readSelfAttention(DataInputStream in, SelfAttention sa) throws IOException {
        Matrix[] wQ = getField(sa, "wQ");
        Matrix[] wK = getField(sa, "wK");
        Matrix[] wV = getField(sa, "wV");
        Matrix   wO = getField(sa, "wO");
        for (Matrix m : wQ) readMatrix(in, m);
        for (Matrix m : wK) readMatrix(in, m);
        for (Matrix m : wV) readMatrix(in, m);
        readMatrix(in, wO);
    }

    private static void writeLayerNorm(DataOutputStream out, LayerNorm ln) throws IOException {
        double[] gamma = getField(ln, "gamma");
        double[] beta  = getField(ln, "beta");
        for (double v : gamma) out.writeDouble(v);
        for (double v : beta)  out.writeDouble(v);
    }

    private static void readLayerNorm(DataInputStream in, LayerNorm ln) throws IOException {
        double[] gamma = getField(ln, "gamma");
        double[] beta  = getField(ln, "beta");
        for (int i = 0; i < gamma.length; i++) gamma[i] = in.readDouble();
        for (int i = 0; i < beta.length;  i++) beta[i]  = in.readDouble();
    }

    private static void writeFeedForward(DataOutputStream out, FeedForwardLayer ffn) throws IOException {
        writeMatrix(out, getField(ffn, "w1"));
        writeMatrix(out, getField(ffn, "w2"));
    }

    private static void readFeedForward(DataInputStream in, FeedForwardLayer ffn) throws IOException {
        readMatrix(in, getField(ffn, "w1"));
        readMatrix(in, getField(ffn, "w2"));
    }

    private static void writeDenseLayer(DataOutputStream out, DenseLayer layer) throws IOException {
        List<Object> neurons = getField(layer, "neurons");
        for (Object neuron : neurons) {
            LinearNeuron ln = (LinearNeuron) neuron;
            com.surish.ai.llm.tensor.Vector weights = getField(ln, "weights");
            double bias = getDoubleField(ln, "bias");
            for (int i = 0; i < weights.size(); i++) out.writeDouble(weights.get(i));
            out.writeDouble(bias);
        }
    }

    private static void readDenseLayer(DataInputStream in, DenseLayer layer) throws IOException {
        List<Object> neurons = getField(layer, "neurons");
        for (Object neuron : neurons) {
            LinearNeuron ln = (LinearNeuron) neuron;
            com.surish.ai.llm.tensor.Vector weights = getField(ln, "weights");
            for (int i = 0; i < weights.size(); i++) weights.set(i, in.readDouble());
            setDoubleField(ln, "bias", in.readDouble());
        }
    }

    private static void writeMatrix(DataOutputStream out, Matrix m) throws IOException {
        for (int i = 0; i < m.rows(); i++)
            for (int j = 0; j < m.columns(); j++)
                out.writeDouble(m.get(i, j));
    }

    private static void readMatrix(DataInputStream in, Matrix m) throws IOException {
        for (int i = 0; i < m.rows(); i++)
            for (int j = 0; j < m.columns(); j++)
                m.set(i, j, in.readDouble());
    }

    @SuppressWarnings("unchecked")
    private static <T> T getField(Object obj, String name) {
        try {
            Class<?> cls = obj.getClass();
            while (cls != null) {
                try {
                    Field f = cls.getDeclaredField(name);
                    f.setAccessible(true);
                    return (T) f.get(obj);
                } catch (NoSuchFieldException e) {
                    cls = cls.getSuperclass();
                }
            }
            throw new RuntimeException("Field not found: " + name);
        } catch (IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    private static Matrix getMatrix(Object obj, String name) {
        return getField(obj, name);
    }

    private static double getDoubleField(Object obj, String name) {
        try {
            Class<?> cls = obj.getClass();
            Field f = cls.getDeclaredField(name);
            f.setAccessible(true);
            return f.getDouble(obj);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static void setDoubleField(Object obj, String name, double value) {
        try {
            Class<?> cls = obj.getClass();
            Field f = cls.getDeclaredField(name);
            f.setAccessible(true);
            f.setDouble(obj, value);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
