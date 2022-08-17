import config
import torch

class BertDataset:

    def __init__(self, review, target):
        self.review = review
        self.target = target

        # Fetch max len and tokenizer from config.py
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        # Return length of dataset
        return len(self.review)

    def __getitem__(self, item):
        # For given item index, return dictionary of inputs
        review = str(self.review[item])
        review = " ".join(review.split())

        inputs = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True
        )

        # IDs are ids of tokens generated after tokenizing
        ids = inputs["input_ids"]

        # Mask is 1 where we have input and 0 where padding
        mask = inputs["attention_mask"]

        # Token type ids behave same as mask
        token_type_ids = inputs["token_type_ids"]

        # Now we return everything
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.target[item], dtype=torch.float)
               }
