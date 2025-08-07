# # train.py
# from torch.optim import AdamW
# from transformers import BertForSequenceClassification, BertTokenizer
# import torch
# from tqdm import tqdm
# from etl import preprocess_data, get_dataloaders

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.backends.cudnn.benchmark = True
# print("Training on device:", device)

# def train():
#     tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#     df = preprocess_data("smart_news_classifier/data/ag_news/train.csv")

#     train_loader, val_loader = get_dataloaders(df, tokenizer)

#     model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
#     model.to(device)
#     optimizer = AdamW(model.parameters(), lr=2e-5)

#     for epoch in range(1):
#         model.train()
#         total_loss = 0
#         for batch in tqdm(train_loader):
#             input_ids, attention_mask, labels = [x.to(device) for x in batch]
#             optimizer.zero_grad()
#             outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#             loss = outputs.loss
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

#     model.save_pretrained("model/")
#     tokenizer.save_pretrained("model/")

# if __name__ == "__main__":
#     train()

# train.py

from torch.optim import AdamW
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch
from tqdm import tqdm
from etl import preprocess_data, get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print("Training on device:", device)

def train():
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    df = preprocess_data("smart_news_classifier/data/ag_news/train.csv")

    train_loader, val_loader = get_dataloaders(df, tokenizer)

    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=4)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(1):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    model.save_pretrained("model/")
    tokenizer.save_pretrained("model/")

if __name__ == "__main__":
    train()


