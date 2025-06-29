import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
class Model(nn.Module):
    def __init__(self,args):
        super(Model, self).__init__()

        self.device = torch.device("cuda")
        self.dropout_rate = args.MODEL.dropout_rate
        self.tokenizer = RobertaTokenizer.from_pretrained(args.MODEL.RobertaPath)
        self.roberta = RobertaModel.from_pretrained(args.MODEL.RobertaPath).to(self.device)

        self.dataset = getattr(args, args.Dataset)
        self.maxlength = getattr(args, args.Dataset).max_length

        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.ie_spv = IE_SPV(dropout_rate=self.dropout_rate).to(self.device)
        self.m_mip = M_MIP(dropout_rate=self.dropout_rate).to(self.device)

    def get_vector(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True,max_length=self.maxlength).to(self.device)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.roberta(**inputs)

        return outputs.last_hidden_state.mean(dim=1)
    
    def get_contextual_embedding(self, sentences, target_words, indices):

        sentences = [s.lower() for s in sentences]
        target_words = [t.lower() for t in target_words]


        inputs = self.tokenizer(sentences, return_tensors='pt', padding=True,max_length=self.maxlength)
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        batch_size = len(sentences)
        all_tokens = [self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][i]) for i in range(batch_size)]
        all_target_indices = []

        for i in range(batch_size):
            tokens_new=all_tokens[i]
            sentence = sentences[i]
            target_index = indices[i]
            words = sentence.split()
            current_token_index = 1
            for j,word in enumerate(words):
                if j==0:
                    new_word_reaining =''.join(self.tokenizer.tokenize(word))
                else:
                    new_word_reaining ='Ġ'+''.join(self.tokenizer.tokenize(word))
                if j==target_index:
                    start_index = current_token_index
                while(new_word_reaining):
                    token = tokens_new[current_token_index]
                    if new_word_reaining.startswith(token):
                        new_word_reaining = new_word_reaining[len(token):]
                    current_token_index+=1
                if j==target_index:
                    end_index = current_token_index
                    all_target_indices.append(list(range(start_index, end_index)))
                    break
        all_target_embeddings = []
        with torch.no_grad():
            outputs = self.roberta(**inputs)
        for i in range(batch_size):
            target_embeddings = outputs.last_hidden_state[i, all_target_indices[i]]

            target_embedding = torch.mean(target_embeddings, dim=0)
            all_target_embeddings.append(target_embedding)



        all_target_embeddings = torch.stack(all_target_embeddings, dim=0)

        all_target_embeddings = self.dropout(all_target_embeddings)

        return all_target_embeddings

    def forward(self, context, expression,Positive_index,NegativeSentence, NegativeExpression, AnchorContext,AnchorExpression,Negative_index):

        positive_target = self.get_vector(expression)
        positive_sentence = self.get_vector(context)
        positive_contextual = self.get_contextual_embedding(context, expression,Positive_index)
        Positive_SPV = self.ie_spv(positive_target, positive_sentence, positive_contextual)
        Positive_MIP = self.m_mip(positive_target, positive_contextual)
        Positive_embedding = torch.cat([Positive_SPV, Positive_MIP], dim=1)


        negative_target = self.get_vector(NegativeExpression)
        negative_sentence = self.get_vector(NegativeSentence)
        negative_contextual = self.get_contextual_embedding(NegativeSentence, NegativeExpression,Negative_index)
        Negative_SPV = self.ie_spv(negative_target, negative_sentence, negative_contextual)
        Negative_MIP = self.m_mip(negative_target, negative_contextual)
        Negative_embedding = torch.cat([Negative_SPV, Negative_MIP], dim=1)

        anchor_target = self.get_vector(AnchorExpression)
        anchor_sentence = self.get_vector(AnchorContext)
        anchor_contextual = self.get_contextual_embedding(AnchorContext,AnchorExpression,Positive_index)
        Anchor_SPV = self.ie_spv(anchor_target, anchor_sentence, anchor_contextual)
        Anchor_MIP = self.m_mip(anchor_target, anchor_contextual)
        Anchor_embedding = torch.cat([Anchor_SPV, Anchor_MIP], dim=1)

        return Positive_embedding, Negative_embedding, Anchor_embedding
    
class IE_SPV(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(IE_SPV, self).__init__()

        self.fc_in = nn.Linear(2 * 768, 768)

        self.dropout = nn.Dropout(dropout_rate)



        self.layer_norm = nn.LayerNorm(768) 

    def forward(self, target, sentence, contextual):
        combined_in = torch.cat([sentence,contextual],dim=1)
        combined_in = self.dropout(combined_in)
        h_in = self.fc_in(combined_in)
    
        h_IE_SPV = self.layer_norm(h_in)
        h_IE_SPV = h_IE_SPV + contextual

        return h_IE_SPV

class M_MIP(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(M_MIP, self).__init__()

        self.fc = nn.Linear(2 * 768, 768) 
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(768) 

    def forward(self, target, contextual):
        combined = torch.cat([contextual, target], dim=1)
        combined = self.dropout(combined)
        output = self.fc(combined)
        output = self.layer_norm(output)
        output = output + target
        return output
# 组合损失函数类
class CombinedLoss(nn.Module):
    def __init__(self,device=torch.device("cuda")):
        super(CombinedLoss, self).__init__()
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_triplet = nn.TripletMarginLoss()

        self.classifier = nn.Sequential(
            nn.Linear(1536*3, 768),
            nn.ReLU(),
            nn.LayerNorm(768),
            nn.Dropout(0.2),
            nn.Linear(768, 2)
        )
        self.classifier = self.classifier.to(device)
        self.criterion_ce = self.criterion_ce.to(device)
        self.criterion_triplet = self.criterion_triplet.to(device)

    def forward(self, Positive_embedding, Negative_embedding, Anchor_embedding, label,device=torch.device("cuda")):
        combined_embedding = torch.cat([Positive_embedding, Negative_embedding, Anchor_embedding], dim=1)
        label = label.to(device)
        logits = self.classifier(combined_embedding)

        loss_ce = self.criterion_ce(logits, label)
        loss_triplet = self.criterion_triplet(Anchor_embedding, Positive_embedding, Negative_embedding)

        loss = loss_ce + loss_triplet
        return loss