import torch

class Tokenizer:
    def __init__(self, vocabulary: dict, pre_tokenizer=' ', special_tokens=[]):
        self.vocabulary = vocabulary
        self.vocabulary_keys = list(vocabulary.keys())
        self.vocabulary_values = list(vocabulary.values())
        self.vocabulary_size = len(vocabulary)
        self.pre_tokenizer = pre_tokenizer
        self.special_tokens = special_tokens

    def encode(self, sequence : str):
        """
        Given a sequence, return the tokenized sequence

        sequence: str, e.g. 'A R N D'
        return: list of int, e.g. [3, 4, 5, 6]
        """
        tokens = sequence.split(self.pre_tokenizer)
        tokens = [self.vocabulary[token] for token in tokens]
        return tokens

    def decode(self, tokens):
        """
        Given a tokenized sequence, return the original sequence

        tokens: list of int, e.g. [3, 4, 5, 6]
        return: str, e.g. 'A B C D'
        """
        sequence = ''
        for token in tokens:
            token_idx = self.vocabulary_values.index(token)
            decoded_token = self.vocabulary_keys[token_idx]
            if decoded_token in self.special_tokens:
                continue
            sequence += decoded_token + self.pre_tokenizer
        return sequence
    
    def get_vocab_size(self):
        return self.vocabulary_size
    
    def to_tensor(self, tokens, dtype=torch.long):
        return torch.tensor(tokens, dtype=dtype).unsqueeze(0)


if __name__ == '__main__':
    # Define the vocabulary
    encoder_vocab = ['PAD', 'SOS', 'EOS', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    decoder_vocab = encoder_vocab + ['SEP']

    # Make a dictionary
    encoder_vocab = {encoder_vocab[i]: i for i in range(len(encoder_vocab))}
    decoder_vocab = {decoder_vocab[i]: i for i in range(len(decoder_vocab))}

    enc_tokenizer = Tokenizer(encoder_vocab, pre_tokenizer=' ', special_tokens=['PAD', 'SOS', 'EOS'])
    dec_tokenizer = Tokenizer(decoder_vocab, pre_tokenizer=' ', special_tokens=['PAD', 'SOS', 'EOS', 'SEP'])

    print(enc_tokenizer.encode('R N D C Q'))
    print(enc_tokenizer.decode([4, 5, 6, 7, 8]))
    print(enc_tokenizer.decode([4, 5, 6, 7, 8, 0, 0, 0]))


    