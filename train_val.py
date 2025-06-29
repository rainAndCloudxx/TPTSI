import torch
import torch.nn as nn
import random
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def set_random_seeds(seed):
    """
    set random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def train_model(args,model, train_loader, val_loader, optimizer, criterion):
    best_f1 = float('-inf')
    dataset = getattr(args, args.Dataset)
    SavePath = dataset.savaPath
    num_epochs = dataset.epochs
    model.train()

    
    for epoch in range(num_epochs):
        total_loss = 0

        for context, expression, label,Positive_index, NegativeSentence, NegativeExpression, AnchorContext,AnchorExpression,Negative_index in train_loader:
        
            optimizer.zero_grad()
            Positive_embedding, Negative_embedding, Anchor_embedding = model(context, expression,Positive_index,NegativeSentence, NegativeExpression,AnchorContext,AnchorExpression,Negative_index)
            loss = criterion(Positive_embedding, Negative_embedding, Anchor_embedding, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("====================")
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')

        accuracy,precision,recall,f1=test_model(model, val_loader, criterion)

        print(f'Test Accuracy: {accuracy * 100}%')
        print(f'Precision: {precision * 100}%')
        print(f'Recall: {recall * 100}%')
        print(f'F1 Score: {f1 * 100}%')

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), SavePath)
            print(f"Best model saved with F1: {best_f1:.4f}")

def test_model(model, test_loader, criterion):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for context, expression, label, Positive_index, NegativeSentence, NegativeExpression, AnchorContext, AnchorExpression, Negative_index in test_loader:
            Positive_embedding, Negative_embedding, Anchor_embedding = model(context, expression, Positive_index, NegativeSentence, NegativeExpression, AnchorContext, AnchorExpression, Negative_index)
            logits = criterion.classifier(torch.cat([Positive_embedding, Negative_embedding, Anchor_embedding], dim=1))
            predicted = torch.argmax(logits, dim=1)

            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())


    processed_labels = []
    processed_predictions = []
    i = 0
    while i < len(all_labels):
        processed_labels.append(all_labels[i])  

        current_pred = all_predictions[i:i + 3]

        weighted_sum = 0.4 * current_pred[0] + 0.3 * current_pred[1] + 0.3 * current_pred[2]

        if weighted_sum >= 0.7:
            processed_predictions.append(1)
        else:
            processed_predictions.append(0)

        i += 3  

    accuracy = accuracy_score(processed_labels, processed_predictions)
    precision = precision_score(processed_labels, processed_predictions)
    recall = recall_score(processed_labels, processed_predictions)
    f1 = f1_score(processed_labels, processed_predictions)

    return accuracy, precision, recall, f1


def MetricCal(args,model,dataLoder,criterion):

    dataset = getattr(args, args.Dataset)
    model_path = dataset.savaPath
    model.load_state_dict(torch.load(model_path))
    print(f"Loaded model weights from {model_path}")
    model.eval()
    accuracy, precision, recall, f1 = test_model(model, dataLoder, criterion)

    # 打印验证结果
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")

