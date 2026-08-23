package com.surish.ai;

import com.surish.ai.llm.corpus.ClasspathCorpusLoader;
import com.surish.ai.llm.corpus.Corpus;
import com.surish.ai.llm.corpus.CorpusLoader;
import com.surish.ai.llm.encoding.CharacterCorpusEncoder;
import com.surish.ai.llm.encoding.CorpusEncoder;
import com.surish.ai.llm.encoding.EncodedCorpus;
import com.surish.ai.llm.model.LanguageModel;
import com.surish.ai.llm.tokenizer.CharacterTokenizer;
import com.surish.ai.llm.tokenizer.Tokenizer;
import com.surish.ai.llm.training.Trainer;
import com.surish.ai.llm.training.TrainingConfig;
import com.surish.ai.llm.vocabulary.CharacterVocabularyBuilder;
import com.surish.ai.llm.vocabulary.Vocabulary;
import com.surish.ai.llm.vocabulary.VocabularyBuilder;

import java.util.List;

public class Main {

    public static void main(String[] args) {

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

        TrainingConfig config = new TrainingConfig(8, 16, 3, 0.01, 0.9);
        LanguageModel model = new LanguageModel(vocabulary.size(), config);
        Trainer trainer = new Trainer(model, encodedCorpus);

        trainer.train();
    }
}
