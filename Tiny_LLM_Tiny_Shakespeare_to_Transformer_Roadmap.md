# Tiny LLM From Scratch — Tiny Shakespeare → Transformer

## Project Goal

We are building a **tiny LLM from scratch in Java**.

Our actual starting point is the **Tiny Shakespeare corpus**, not an artificial example.

The purpose is to understand the complete path from raw text to a trained language model:

```text
Tiny Shakespeare
      ↓
Tokenization
      ↓
Vocabulary
      ↓
Token IDs
      ↓
Token Embeddings
      ↓
Neural Network
      ↓
Next-Token Prediction
      ↓
Loss
      ↓
Gradient
      ↓
Backpropagation
      ↓
Parameter Updates
      ↓
Training
      ↓
Context
      ↓
Attention
      ↓
Transformer
      ↓
Text Generation
```

The final goal is a **working tiny Transformer-based language model**, trained on Tiny Shakespeare.

---

# IMPORTANT: Where We Are Right Now

We have already started the implementation.

Our current run shows:

```text
Tokens: 1115394
Vocabulary: 65 characters
Embedding dimension: 16
Dense output: 8 values
Target: 8 values
```

We have already demonstrated:

```text
Token
  ↓
Vocabulary ID
  ↓
Embedding vector
  ↓
Dense layer
  ↓
Output
  ↓
MSE Loss
  ↓
Gradient
  ↓
Parameter update
  ↓
Output changes
  ↓
Loss decreases
```

Example from the current implementation:

```text
Loss BEFORE
0.1280876620545637

Loss AFTER
0.1274450486864671
```

So the first important learning loop is already working.

We should **continue from here**, not restart the project.

---

# PART 1 — WHAT WE HAVE ALREADY BUILT

## Stage 1 — Tiny Shakespeare Corpus

### What we did

We loaded the Tiny Shakespeare text corpus.

The current corpus contains approximately:

```text
1.1 million characters/tokens
```

Because our current tokenizer is character-based, the tokens are characters.

### Why we did it

A language model needs actual training data.

Tiny Shakespeare gives us a real body of text containing:

- words
- punctuation
- spaces
- repeated patterns
- dialogue
- sentence structure
- longer-range relationships

This gives the model something real to learn from.

---

# Stage 1 — Character Tokenization

### What we did

We currently tokenize the corpus at the character level.

For example:

```text
India
```

becomes:

```text
I
n
d
i
a
```

For Tiny Shakespeare, the vocabulary contains the characters that occur in the corpus.

### Why we did it

The neural network cannot directly process:

```text
"To be, or not to be"
```

It needs numerical representations.

Tokenization defines the basic units that the model works with.

We are deliberately starting with characters because it keeps the first implementation understandable.

---

# Stage 2 — Vocabulary

### What we did

We created a mapping between each character/token and an integer ID.

Our current vocabulary has:

```text
65 tokens
```

Conceptually:

```text
A → 13
B → 14
C → 15
...
a → 39
b → 40
...
```

There is also the reverse mapping:

```text
13 → A
14 → B
...
```

### Why we did it

The vocabulary gives every token a stable numerical identity.

The important distinction is:

```text
Token
  = actual character/token

Vocabulary
  = collection + mapping

Token ID
  = numerical identity used by the model
```

The token ID itself is **not the learned meaning** of the token.

It is an index.

---

# Stage 3 — Token IDs → Embedding

### What we did

We created an embedding representation.

Our current experiment uses:

```text
Embedding dimension = 16
```

So a token is represented by 16 numerical values.

Conceptually:

```text
Token ID
   ↓
Embedding table
   ↓
16 values
```

For example:

```text
token ID 32
      ↓
[0.01, -0.02, 0.03, ...]
```

### Why we did it

A token ID such as:

```text
32
```

is just an index.

The network needs a vector representation that can participate in mathematical operations.

The embedding gives each vocabulary token a trainable vector.

### Important understanding

There is not one new embedding for every occurrence of a token.

If the same token appears 10,000 times, it still refers to the same vocabulary embedding row.

---

# Stage 4 — Dense Layer

### What we did

We passed the embedding through a Dense layer.

Current experiment:

```text
Embedding
   ↓
Dense layer
   ↓
8 output values
```

### Why we did it

The embedding is only a representation.

The Dense network transforms that representation into another representation.

A Dense layer contains its own learned parameters:

```text
weights
biases
```

This is important because the embedding table is **not the whole model**.

The network weights are learned parameters too.

---

# Stage 5 — Prediction

### What we did

The Dense layer produces an output.

At this early stage the output is just a vector of numbers.

### Why we did it

The model needs to produce something that can be compared against a target.

This gives us the basic structure:

```text
input
 ↓
embedding
 ↓
network
 ↓
prediction
```

---

# Stage 6 — MSE Loss

### What we did

We currently use Mean Squared Error.

Conceptually:

```text
prediction
     ↓
compare with target
     ↓
calculate difference
     ↓
square differences
     ↓
average
     ↓
loss
```

### Why we used MSE

At this stage we are learning the mechanics of neural-network training.

MSE is simple enough that we can directly understand:

- prediction
- error
- loss
- gradient
- parameter update

It is an educational stepping stone.

We will later move to the loss normally used for next-token classification.

---

# Stage 7 — Loss Gradient

### What we did

We calculated the gradient of the MSE loss.

The current output shows:

```text
Gradient
[-0.2526, ...]
```

### Why we did it

The loss tells us:

> How wrong was the prediction?

The gradient gives us information about:

> Which direction should the values move to reduce the loss?

This is what allows the network to learn rather than simply calculate outputs.

---

# Stage 8 — Parameter Update

### What we did

We used the gradient to update parameters.

We observed:

```text
Output BEFORE
        ↓
Gradient
        ↓
Output AFTER
```

and:

```text
Loss BEFORE
0.128...

Loss AFTER
0.127...
```

### Why this matters

This is our first real demonstration of learning.

The network changed its parameters based on the error.

The output changed.

The loss decreased.

That is the basic learning mechanism that will remain even when we eventually reach Transformers.

---

# Stage 9 — Gradient Flow Back Into the Embedding

## What we will do next

Our current training flow is:

```text
T
 ↓
Token ID
 ↓
T embedding
 ↓
Dense layer
 ↓
Output
 ↓
MSE Loss
 ↓
Gradient
```

We already calculate the gradient needed to update the Dense layer.

The next step is to continue that gradient backward through the Dense layer and calculate how much the **input embedding contributed to the loss**.

The flow becomes:

```text
Loss
 ↓
Dense gradient
 ↓
Embedding gradient
```

Then the parameters used by this example are updated:

```text
Dense weights  → updated
T embedding    → updated
```

## Why we need this

Right now, the Dense layer can learn from the error, but the embedding also needs to become a learned representation.

The embedding is a trainable part of the model, not just a fixed input.

For the current example using `T`, the gradient should reach the embedding row corresponding to `T`.

We are **not** updating every vocabulary embedding for this single example.

Conceptually:

```text
T
 ↓
T embedding
 ↓
Dense
 ↓
prediction
 ↓
loss
 ↓
backpropagation
 ↓
T embedding gradient
 ↓
T embedding update
```

## What we should verify

After this stage, we should be able to observe that:

```text
T embedding BEFORE
        ↓
training step
        ↓
T embedding AFTER
```

and:

```text
Dense weights BEFORE
        ↓
training step
        ↓
Dense weights AFTER
```

We should also verify that the loss decreases after the update.

The important result of this stage is:

> **The gradient can flow through the Dense layer back into the embedding, allowing both the Dense parameters and the `T` embedding to learn from the same error.**

# Stage 10 — Understand What Is Stored

### What we established

After training, the model does not consist only of embeddings.

The trained model contains multiple sets of learned parameters.

Conceptually:

```text
Model
├── Vocabulary
├── Token embeddings
├── Dense/network weights
├── Biases
└── Later: Transformer parameters
```

### Why this matters

This resolves an important concept we discussed.

The relationship between tokens is not stored as a separate database of facts.

Instead, training changes numerical parameters.

Those learned parameters collectively encode patterns that the network can use during inference.

The embedding parameters are one part.

The network parameters are another part.

---

# PART 2 — TURNING OUR CURRENT NETWORK INTO A LANGUAGE MODEL

The current experiment proves that our network can learn to reduce a loss.

It is **not yet a proper language model**.

The next goal is to make the prediction target come directly from Tiny Shakespeare.

---

# Stage 11 — Create Real Next-Token Training Examples

## What we will do

Instead of manually creating a target vector, use the actual text.

For example:

```text
hello
```

creates training relationships such as:

```text
h → e
e → l
l → l
l → o
```

For Tiny Shakespeare:

```text
T → h
h → e
e → space
space → b
b → e
...
```

### Why we need this

A language model's fundamental task is:

> Given previous token(s), predict the next token.

This changes our current experiment from:

```text
arbitrary vector → arbitrary target
```

into:

```text
text context → actual next token
```

That is the critical transition into language modeling.

---

# Stage 12 — Make the Output Vocabulary-Sized

## What we will do

Our current output has only:

```text
8 values
```

That was useful for learning the mechanics.

For language modeling, if our vocabulary has:

```text
65 tokens
```

the final prediction must produce:

```text
65 scores
```

One score for every possible next token.

Conceptually:

```text
input
 ↓
network
 ↓
65 scores
```

### Why we need this

The model must be able to answer:

> Which of the 65 possible tokens should come next?

---

# Stage 13 — Logits

## What we will do

The final layer will produce raw scores called logits.

Example:

```text
A → 1.2
B → -0.4
C → 3.7
D → 0.2
...
```

### Why we need this

The network naturally produces numerical scores.

These scores will later be converted into probabilities.

The largest score indicates which token the model currently favors.

---

# Stage 14 — Softmax

## What we will do

Convert logits into probabilities.

```text
Logits
 ↓
Softmax
 ↓
Probability for every vocabulary token
```

Example:

```text
A → 0.05
B → 0.01
C → 0.82
D → 0.12
```

### Why we need this

Now the model can express a probability distribution over the next token.

This gives us:

```text
P(next token | previous tokens)
```

which is the central prediction made by a language model.

---

# Stage 15 — Cross-Entropy Loss

## What we will do

Replace the educational MSE objective with cross-entropy for next-token prediction.

```text
logits
 ↓
softmax
 ↓
probabilities
 ↓
cross-entropy
 ↓
loss
```

### Why we need this

Next-token prediction is effectively a classification problem:

```text
Choose 1 token
from the entire vocabulary
```

Cross-entropy is designed for this type of probability prediction.

We will understand its gradient rather than treating the formula as magic.

For softmax + cross-entropy, the gradient simplifies to:

```text
predicted probabilities - target
```

---

# Stage 16 — Train on Tiny Shakespeare

## What we will do

Now the complete basic language-model training loop becomes:

```text
Tiny Shakespeare
      ↓
tokens
      ↓
token IDs
      ↓
embeddings
      ↓
network
      ↓
logits
      ↓
softmax
      ↓
next-token prediction
      ↓
cross-entropy loss
      ↓
gradient
      ↓
backpropagation
      ↓
update embeddings + network weights
      ↓
repeat
```

### Why we need this

At this point we will have a real, though very limited, character-level language model.

It will actually learn patterns from Tiny Shakespeare.

---

# PART 3 — GIVE THE MODEL CONTEXT

The current model can begin with:

```text
one token → next token
```

That is far too limited.

We now need to teach it to use a sequence of previous tokens.

---

# Stage 17 — Context Window

## What we will do

Instead of:

```text
T → ?
```

we provide:

```text
To be → ?
```

or:

```text
To be, or → ?
```

depending on the chosen context length.

### Why we need this

The next token depends on more than one previous token.

For example:

```text
The king of
```

contains much more information than:

```text
of
```

A language model needs context to make useful predictions.

---

# Stage 18 — Multiple Token Embeddings

## What we will do

A sequence gives us multiple embeddings:

```text
token 1 → embedding 1
token 2 → embedding 2
token 3 → embedding 3
token 4 → embedding 4
```

We need to combine these representations.

Initially we can experiment with simple approaches such as:

```text
concatenation
average
sum
```

### Why we need this

This lets us directly see the problem that later motivates attention.

The model must preserve useful information from multiple tokens.

---

# Stage 19 — Position

## What we will do

We introduce positional information.

For example:

```text
A B C
```

is different from:

```text
C B A
```

even though the same tokens exist.

### Why we need this

The model needs to know:

```text
what token
+
where it occurs
```

This becomes an essential part of Transformer architecture.

---

# PART 4 — ATTENTION

Now we reach the major architectural change.

---

# Stage 20 — Understand the Problem Attention Solves

## What we will do

We will take a sequence and ask:

> When processing one token, which other tokens should matter?

For example:

```text
The king spoke to the queen because she was angry.
```

The model may need to connect:

```text
she
```

with:

```text
queen
```

rather than treating every token equally.

### Why we need this

Simple averaging or concatenation does not provide a flexible mechanism for deciding which tokens are important to each other.

Attention provides that mechanism.

---

# Stage 21 — Query, Key, Value

## What we will do

For each token representation, create:

```text
Query
Key
Value
```

Then:

```text
Query
  ↓
compare with Keys
  ↓
attention scores
  ↓
softmax
  ↓
attention weights
  ↓
weighted Values
```

### Why we need this

This allows one token to selectively use information from other tokens.

The model learns the transformations that produce useful queries, keys, and values.

---

# Stage 22 — Self-Attention

## What we will do

Apply attention among tokens in the same sequence.

```text
Token 1 ─┐
Token 2 ─┤
Token 3 ─┼→ Self-Attention
Token 4 ─┤
Token 5 ─┘
```

### Why we need this

Now the representation of a token can depend on the other tokens in its context.

This is the key mechanism that gives Transformers their powerful contextual representation.

---

# Stage 23 — Causal Mask

## What we will do

For language generation, a token must not look at future tokens.

Example:

```text
The cat sat ...
```

When predicting the next token, the model must not see the actual future answer.

We therefore apply a causal mask.

### Why we need this

Training must simulate the real generation situation.

During generation, the future does not exist yet.

---

# Stage 24 — Multi-Head Attention

## What we will do

Instead of one attention mechanism, create multiple attention heads.

```text
Input
 ├── Head 1
 ├── Head 2
 ├── Head 3
 └── Head 4
       ↓
    combine
```

### Why we need this

Different heads can learn different relationships in the sequence.

This gives the model several parallel ways to examine contextual relationships.

---

# PART 5 — BUILD THE TRANSFORMER

---

# Stage 25 — Feed-Forward Network

## What we will do

After attention, pass each token representation through a small neural network.

```text
attention output
      ↓
linear layer
      ↓
activation
      ↓
linear layer
```

### Why we need this

Attention allows information to move between tokens.

The feed-forward network performs additional learned transformations on each token representation.

Both are important parts of a Transformer block.

---

# Stage 26 — Residual Connections

## What we will do

Add skip/residual connections around major sublayers.

Conceptually:

```text
input ──────────────┐
 ↓                  +
attention ──────────┘
```

### Why we need this

As we stack more layers, information and gradients need effective paths through the network.

Residual connections help make deep networks trainable.

---

# Stage 27 — Normalization

## What we will do

Add normalization around Transformer sublayers.

### Why we need this

Deep networks repeatedly transform representations.

Normalization helps keep those representations numerically well behaved and makes training more stable.

---

# Stage 28 — One Transformer Block

## What we will do

Combine:

```text
Input
 ↓
Self-Attention
 ↓
Residual + Normalization
 ↓
Feed-Forward Network
 ↓
Residual + Normalization
 ↓
Output
```

### Why we need this

This becomes the basic reusable unit of our Transformer.

---

# Stage 29 — Stack Transformer Blocks

## What we will do

Instead of one block:

```text
Input
 ↓
Block
 ↓
Output
```

we build:

```text
Input
 ↓
Block 1
 ↓
Block 2
 ↓
Block 3
 ↓
...
 ↓
Block N
```

### Why we need this

More layers allow the model to progressively transform its representations.

This is where our simple neural network becomes a deeper Transformer architecture.

---

# PART 6 — COMPLETE TINY LLM

---

# Stage 30 — Transformer Output

## What we will do

Take the final Transformer representation and project it to the vocabulary.

```text
Transformer
 ↓
final representation
 ↓
output projection
 ↓
65 vocabulary logits
```

### Why we need this

The Transformer produces useful contextual representations.

The final projection converts those representations into actual next-token scores.

---

# Stage 31 — Train the Complete Transformer

## What we will do

Now everything connects:

```text
Tiny Shakespeare
      ↓
Token IDs
      ↓
Token Embeddings
      ↓
Position
      ↓
Transformer
      ├── Self-Attention
      ├── Feed-Forward
      ├── Residuals
      └── Normalization
      ↓
Vocabulary logits
      ↓
Softmax
      ↓
Next-token probabilities
      ↓
Cross-Entropy
      ↓
Gradient
      ↓
Backpropagation
      ↓
Update ALL learned parameters
```

The learned parameters include:

```text
Embedding parameters
Attention weights
Feed-forward weights
Normalization parameters
Output weights
Biases where applicable
```

### Why we need this

This is the actual end-to-end training of our tiny LLM.

The learning mechanism is still the same one we learned at the beginning:

```text
Forward
 ↓
Loss
 ↓
Gradient
 ↓
Backpropagation
 ↓
Update
```

The major difference is that the network through which the information flows is now a Transformer.

---

# Stage 32 — Generate Shakespeare-like Text

## What we will do

After training, give the model a prompt:

```text
ROMEO:
```

The model:

```text
prompt
 ↓
tokens
 ↓
embeddings
 ↓
Transformer
 ↓
logits
 ↓
probabilities
 ↓
select next token
 ↓
append token
 ↓
run again
 ↓
repeat
```

### Why we need this

Training teaches:

```text
predict the next token
```

Generation repeatedly uses that capability.

This turns next-token prediction into a generated sequence of text.

---

# Stage 33 — Save and Load the LLM

## What we will do

Save the trained model.

Conceptually:

```text
tiny-llm-model
├── vocabulary
├── embedding parameters
├── Transformer weights
├── output weights
├── biases
└── configuration
```

Then:

```text
Train
 ↓
Save
 ↓
Stop Java program
 ↓
Start Java program
 ↓
Load parameters
 ↓
Generate text
```

### Why we need this

This proves that the learned model is represented by its trained parameters.

We do not need to retrain every time we want to use it.

---

# Stage 34 — Evaluate and Improve

## What we will do

Measure:

- training loss
- validation loss
- next-token accuracy
- generated samples
- effect of context length
- effect of embedding dimension
- effect of number of layers
- effect of attention heads
- effect of learning rate

### Why we need this

We want to understand which architectural and training choices actually improve the model.

---

# FINAL PROJECT ARCHITECTURE

Our completed project should look approximately like this:

```text
                    Tiny Shakespeare
                           │
                           ▼
                    CharacterTokenizer
                           │
                           ▼
                       Vocabulary
                           │
                           ▼
                       Token IDs
                           │
                           ▼
                    Token Embeddings
                           │
                           ▼
                  Positional Information
                           │
                           ▼
                 ┌─────────────────────┐
                 │   Transformer 1     │
                 │                     │
                 │ Self-Attention      │
                 │ Feed-Forward        │
                 │ Residuals           │
                 │ Normalization       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Transformer 2     │
                 └──────────┬──────────┘
                            │
                           ...
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Transformer N     │
                 └──────────┬──────────┘
                            │
                            ▼
                    Output Projection
                            │
                            ▼
                   Vocabulary Logits
                            │
                            ▼
                         Softmax
                            │
                            ▼
                 Next-Token Probabilities
                            │
                            ▼
                   Cross-Entropy Loss
                            │
                            ▼
                       Gradients
                            │
                            ▼
                    Backpropagation
                            │
                            ▼
              Update All Model Parameters
                            │
                            └───────────────┐
                                            │
                                            ▼
                                      Repeat Training
```

---

# THE IMPORTANT CONTINUITY

The project should always be understood as one continuous progression.

We are **not building unrelated projects**.

We are progressively replacing a simple part with a more capable part.

```text
CURRENT
Embedding → Dense → MSE
```

becomes:

```text
Embedding → Dense → Vocabulary Prediction → Cross-Entropy
```

then:

```text
Context → Embeddings → Neural Network → Next Token
```

then:

```text
Context → Attention → Next Token
```

then:

```text
Context
   ↓
Multi-Head Self-Attention
   ↓
Feed-Forward
   ↓
Transformer Blocks
   ↓
Next Token
```

and finally:

```text
Tiny Shakespeare
      ↓
Tokenizer
      ↓
Embeddings
      ↓
Transformer
      ↓
Next-token prediction
      ↓
Training
      ↓
Generation
```

---

# CORE CONCEPTS WE HAVE ALREADY ESTABLISHED

These should remain clear throughout the project.

## Token vs Vocabulary

```text
Token = one actual token

Vocabulary = collection of all known tokens
```

## Token ID

```text
Token
  ↓
Vocabulary lookup
  ↓
Token ID
```

The ID is an index.

## Embedding

```text
Token ID
  ↓
Embedding table
  ↓
Vector
```

There is one learned embedding vector for each vocabulary token.

## Network Weights

The Dense/Transformer network has its own learned parameters.

They are separate from the embedding table.

```text
Model parameters
├── Embedding parameters
└── Network parameters
```

## Training

Training changes those parameters.

```text
Prediction
 ↓
Loss
 ↓
Gradient
 ↓
Backpropagation
 ↓
Update parameters
```

Repeated over a large amount of data, this causes the parameters to encode useful statistical relationships from the training corpus.

## Inference

After training:

```text
Prompt
 ↓
Token IDs
 ↓
Embeddings
 ↓
Transformer
 ↓
Output probabilities
 ↓
Next token
```

The model does not look up a stored answer sentence.

It computes the next-token probabilities using the learned parameters.

---

# FINAL MILESTONES

## Already completed

- [x] Tiny Shakespeare corpus loaded
- [x] Character tokenizer
- [x] Vocabulary
- [x] Token IDs
- [x] Embedding
- [x] Dense layer
- [x] Forward pass
- [x] Target
- [x] MSE
- [x] MSE gradient
- [x] Parameter update
- [x] Demonstrated loss reduction

## Next

- [ ] Connect training examples directly to Tiny Shakespeare
- [ ] Make prediction target the actual next token
- [ ] Make output size equal vocabulary size
- [ ] Logits
- [ ] Softmax
- [ ] Cross-entropy
- [ ] Full next-token training loop
- [ ] Train embeddings and network together
- [ ] Context window
- [ ] Multiple-token input
- [ ] Positional information
- [ ] Query / Key / Value
- [ ] Self-attention
- [ ] Causal masking
- [ ] Multi-head attention
- [ ] Feed-forward network
- [ ] Residual connections
- [ ] Normalization
- [ ] Transformer block
- [ ] Multiple Transformer blocks
- [ ] Output projection
- [ ] End-to-end Transformer training
- [ ] Text generation
- [ ] Save/load trained model
- [ ] Evaluation
- [ ] Tiny Shakespeare Transformer LLM

---

# THE FINAL GOAL

At the end, we should be able to run something conceptually like:

```text
Prompt:
ROMEO:

Tiny LLM:
ROMEO: ...
```

The important achievement is not that the generated Shakespeare is perfect.

The important achievement is that **we built the entire learning mechanism ourselves**:

```text
raw text
→ tokens
→ vocabulary
→ embeddings
→ neural network
→ prediction
→ loss
→ gradient
→ backpropagation
→ learned parameters
→ context
→ attention
→ Transformer
→ next-token prediction
→ generation
```

That is the complete path we are following to build our Tiny LLM.
