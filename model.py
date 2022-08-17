import config
import transformers
import torch.nn as nn

class BERTBaseUncased(nn.Module):

    def __init__(self):
        super(BERTBaseUncased, self).__init__()

        # Fetch the model from BERT_PATH in config
        self.bert = transformers.BertModel.from_pretrained(
            config.BERT_PATH
        )

        # Add dropout for regularization
        self.bert_drop = nn.Dropout(0.3)

        # A simple linear layer for output
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):

        # In its default state, BERT returns two outputs:
        # the last hidden state and output of BERT pooler layer
        # We use the output of the pooler which is of the size
        # (batch_size, hidden_size).  hidden_size can be 768 or 1024
        # depending on if we are using BERT base or large.  In our case,
        # it is 768
        _, o2 = self.bert(ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids
                          )
        # Pass through dropout layer
        bo = self.bert_drop(o2)

        # Pass through linear layer
        output = self.out(bo)

        return output 
