package com.surish.ai;

import com.surish.ai.llm.corpus.ClasspathCorpusLoader;
import com.surish.ai.llm.corpus.Corpus;
import com.surish.ai.llm.corpus.CorpusLoader;
import com.surish.ai.llm.encoding.CharacterCorpusEncoder;
import com.surish.ai.llm.encoding.CorpusEncoder;
import com.surish.ai.llm.encoding.EncodedCorpus;
import com.surish.ai.llm.model.LanguageModel;
import com.surish.ai.llm.model.ModelSerializer;
import com.surish.ai.llm.tokenizer.CharacterTokenizer;
import com.surish.ai.llm.tokenizer.Tokenizer;
import com.surish.ai.llm.training.Trainer;
import com.surish.ai.llm.training.TrainingConfig;
import com.surish.ai.llm.vocabulary.CharacterVocabularyBuilder;
import com.surish.ai.llm.vocabulary.Vocabulary;
import com.surish.ai.llm.vocabulary.VocabularyBuilder;

import java.io.File;
import java.util.List;

public class Main {

    private static final String MODEL_PATH = "model.bin";

    public static void main(String[] args) throws Exception {

        CorpusLoader loader = new ClasspathCorpusLoader("tiny_shakespeare.txt");
        Corpus corpus = loader.load();

        Tokenizer<Character> tokenizer = new CharacterTokenizer();
        List<Character> tokens = tokenizer.tokenize(corpus);

        VocabularyBuilder<Character> builder = new CharacterVocabularyBuilder();
        Vocabulary<Character> vocabulary = builder.build(tokens);

        System.out.println("Tokens: " + tokens.size());
        System.out.println("Vocabulary size: " + vocabulary.size());

        CorpusEncoder<Character> encoder = new CharacterCorpusEncoder();
        EncodedCorpus encodedCorpus = encoder.encode(tokens, vocabulary);

        TrainingConfig config = new TrainingConfig(32, 64, 4, 8, 30, 0.0001, 0.9);
        LanguageModel model = new LanguageModel(vocabulary.size(), config);

        if (new File(MODEL_PATH).exists()) {
            System.out.println("Loading saved model from " + MODEL_PATH);
            ModelSerializer.load(model, MODEL_PATH);
        } else {
            Trainer trainer = new Trainer(model, encodedCorpus);
            trainer.train();
            System.out.println("Saving model to " + MODEL_PATH);
            ModelSerializer.save(model, MODEL_PATH);
        }

        System.out.println("\n--- Prediction ---\n");
        System.out.println(model.predict("F", 200, vocabulary));
    }
}
