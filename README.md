# NLP-Machine-Translation-Using-Deep-Learning

I develop a deep neural network in this project, which operates as part of a machine pipeline. The input to the input is English and the translation is returned in French. The aim is to get the maximum feasible precision in translation. 

An essential part of being human is the ability to communicate with one another. Worldwide there are around 7,000 languages. Language translation is a key cultural and economic link between individuals from different nations and ethnic groups as our globe gets more and more connected. 

Some of the more evident applications include: 

Business: worldwide commerce, investment, contracts, financing. 

commerce: travel, overseas products and services purchases, customer support. 

Media: search information, social network information exchange, content location, and advertising. media: information access. 

Education: ideas sharing, teamwork, research paper translation. 

Government: international relations, negotiations. 

Tech companies spend extensively in machine translation to address these demands. This investment and recent advances in profound education have brought significant increases in the quality of translation. The translation accuracy was 60% higher, according to Google, than the phrase-based methodology previously utilized in Google Translate. Today, Google and Microsoft can translate more than 100 different languages and several of them get closer to human level precision. 

Although machine translation has made many advances, it is not yet perfect. 

I have to establish a recurrent neural network to translate a corpus of English into French (RNN). Let's start by introducing RNN's and why they're beneficial for NLP jobs before plunging into the implementation. 

RNN Overview: 

The RNNs are designed to accept text sequences as inputs or text sequences, or both. They are referred to as recurrence as the hidden layers of the network are looped in by the output and cell state at the following stage. It is a type of memory. This repetition. It permits contextual information to flow across the network so that the appropriate outputs from past processes may be applied at the current time to network activities. 

The way I interpret this is similar. You save crucial data from previous words and phrases when reading this post and use it as background for understanding each new word and sentence. 

This cannot be done by other kinds of neural networks (yet). Imagine using a convolutionary neural network (CNN) to recognize objects in a film. Currently, information from items identified in past scenes cannot be sent to the model in the present scene for the detection of items. If, for instance, in an earlier scene a judge and court chamber were recognized, this information may properly identify the gavel in the current scene rather than misclassify it as a hammer or briefing. But CNNs are not permitted to flow through the network like RNNs by this sort of time series environment. 

RNN Setup 

You want to configure your RNN to manage inputs and outputs differently depending on the circumstance. I will utilize a many-to-many process for this project where input is an English word sequence and the output is a French word sequence (fourth from the left in the diagram below).

![sequence](https://user-images.githubusercontent.com/96385070/148429424-a0254af3-681a-453b-9f48-ec4465182bd7.png)

The arrows are a vector and indicate functions in each rectangle (e.g. matrix multiply). The input vectors are in red, the output vectors are in blue, and the RNN is in green (more on this soon). 

From left to right: from left to right: (1) RNN-free vanilla processing mode, from the fixed input to the fixed output (e.g. image classification). (2) Output of sequence (e.g. image captioning takes an image and outputs a sentence of words). (3) Input of sequence (e.g. sentiment analysis where a given sentence is classified as expressing positive or negative sentiment). (4) Input and output sequence (e.g. Machine Translation: an RNN reads a sentence in English and then outputs a sentence in French). (5) Input and output synchronized sequence (e.g. video classification where I wish to label each frame of the video). Note that in either instance, the length sequences do not have any pre-specified restriction because the recurrent (green) transformation is fixed and may be carried out as many as I choose. 

 

Building the Pipeline: 

Preprocessing: loading and testing of information, purification, tokenization, padding 

Modeling: construction, railway, and model testing 

Prediction: create unique(specific) English to French translations, then compare the results to the basic true interpretations 

Iteration: model iterate, experiment with various structures. 

Frameworks: 

For the frontend, I utilize Keras and for the back end of the project, TensorFlow. I like to use Keras over the TensorFlow since the syntax is simpler that makes it more straightforward to create the model layers. But you lose the ability to make sophisticated adaptations using Keras. But that will not influence the models in this project we are constructing. 

Preprocessing: 

Load and Review Data . 

A sample of the data is provided here. The inputs are in English; the outputs are translated into French.

![trasnlate](https://user-images.githubusercontent.com/96385070/148429755-34219a59-3321-4c10-b63d-1535ee8bae03.JPG)

If I count a word, I can observe that the dataset vocabulary is really small. This was the project's design. This gives us a decent time to train the models.

![Count](https://user-images.githubusercontent.com/96385070/148429909-23a10e53-f699-4eef-b60a-7aa6e054ea09.JPG)

Cleaning: 

At this time, no more cleaning is necessary. The data has been transformed and separated such that spaces between all words and punctuation are available. 

Tokenization: 

Next, the data should be tokenized — i.e. the text converted to numerical values. This makes it possible to operate on the input data on the neural network. For this project, a single ID is provided for each word and punctuation mark. (To assign each character a unique ID would be useful for other NLP studies). 

When the tokenizer is executed, it will generate a word index, and each phrase may be converted to a vector.


Padding: 

Each sequence must have the same length when feeding our sequences of word IDs into the model. To do this, each sequence shorter than the max length is added to padding (i.e. shorter than the longest sentence). 

![pad](https://user-images.githubusercontent.com/96385070/148430205-ec0b8357-1376-4982-a81f-4f5ea9f18241.JPG)

Modeling: 

Let's first unravel the high-level RNN architecture. There are several components of the model that I should be aware of in the figure above: 

Inputs: Each step is supplied into the model using a single word. Each word is encrypted as a single integer or one-hot encoded vector that corresponds to the vocabulary of English data set. 

Embedding Layers: Incorporations are used to turn every word to a vector. The vector size is influenced by the vocabulary's complexity. 

Recurrent Layers (Encoder): This provides the context for the current word vector from word vectors in earlier time steps. 

Dense layers: These layers are usual for decoding the encoded input to the right translations sequence and are fully linked. 

Output: the outputs are returned to the vocabulary of the French dataset as an integer or one-hot encoded vector sequence. 

Embeddings: 

Embeddings allow us to record syntactic and semantic word connections more accurately. This is achieved when each word is projected into an n-dimensional place. The likelihood of terms in this area is that the closer two words are, the more similar they are. And the vectors of words frequently mean meaningful interactions, such as gender, heated verbal interactions or even geopolitical linkages.

![emb](https://user-images.githubusercontent.com/96385070/148431009-b3110f23-19f0-41a5-b73e-8c019f52f328.JPG)

Training embeddings from scratch on a big dataset requires a great deal of information and calculation. So, I usually used a pre-trained embedding package, like Glove or word2vec, instead of doing it ourselves. When this is used, embedding is a form of transfer learning. However, because I have a low vocabulary and small syntactic change in our dataset for this project, I will use Keras to train our own embeddings. 

Encoder & Decoder: 

Our sequential architecture connects two recurring networks: an encoder and a decoder. The encoder summarizes the input to a context variable, often known as status. This context is then decoded, and the sequence of output is produced.

![pic](https://user-images.githubusercontent.com/96385070/148431624-e62236da-94c9-4aef-b7db-033332fc38a1.JPG)

Since the encoder and decoder are recurring, each section of the sequence has loops, which are processed at various times. It is ideal to unroll the network to image that, so that I can see what is going on every single step. 

It takes four times to encode the whole input sequence in the following example. The encoder "reads" the input word each time and transforms its hidden state. Then the concealed state advances to the next stage. Bear in mind that the hidden state is the context in which the network flows. The more the hidden state, the larger is the model's learning capability, but the higher the calculation needs. When I discuss gated recurring units, I will discuss more about the changes in the concealed state (GRU).

![enc](https://user-images.githubusercontent.com/96385070/148431909-ca2d13e7-a795-425e-ae6c-36a6b9d2f556.JPG)

For now, note that two entries are available for each step following the first word in the sequence: the concealed state and a sequence text. This is the next word for the encoder in the series of entries. This is the preceding word in the output sequence for the decoder. 

Recall also that when I refer to a word, the term that comes from the embedding layer really refers to a vector representation. 

Bidirectional Layer: 

Now that I understand how the network is flowing via the hidden state, let us go further, enabling this context to flow in both directions. That's what a bidirectional accomplishes. The encoder has simply a historical background in the given scenario. However, higher model performance might arise from the future context. This may appear contrary to the way people process language as I only read in one way. People, however, frequently need to understand what is being stated in future contexts. In other words, sometimes a statement is incomprehensible until a key word or sentence is given. To do this, I are concurrently training two RNN layers. The first layer provides a reverse replica of the input sequence and the second layer. 

![GRU](https://user-images.githubusercontent.com/96385070/148432380-a014f34b-02f9-45a8-9e77-0bbf5b5f364c.JPG)

Hidden Layer with Gated Recurrent Unit (GRU): 

Now, let's slightly wiser our RNN. What if I could be more selective rather than permit all the information from the concealed state to pass through the network? Some facts could be more pertinent, while others should be dismissed. This is mostly done through a guarded recurring unit (GRU). 

Two gates in the GRU are available: the update door and the reset door. To summarize, the update gate (z) assists the model to calculate the amount of information from earlier stages into the future. The reset gate (r) decides, however, how much the knowledge from the past should be forgotten. 

![asd](https://user-images.githubusercontent.com/96385070/148432548-f9dbe25f-9b05-4a2d-a5d2-e29fe3c7d603.JPG)

![accuracy](https://user-images.githubusercontent.com/96385070/148432652-50e79e1d-0ea2-4c29-bde4-57c283d0940d.JPG)

# The Final accuracy we achieved is 95.9%
