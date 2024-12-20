import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

print("Загрузка данных...")
data = pd.read_csv('C:/Users/vlaados/Desktop/tip/dataset.csv')
texts = data['text'].tolist()
labels = data['label'].tolist()
print(f"Данные загружены. Всего записей: {len(texts)}")

print("Разделение данных на обучающую и тестовую выборки...")
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
print(f"Тренировочная выборка: {len(train_texts)} записей, Тестовая выборка: {len(val_texts)} записей")

print("Инициализация токенайзера и модели...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=9)
model.to(device)
print("Токенайзер и модель загружены и перемещены на устройство.")

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

print("Создание DataLoader для тренировочной и тестовой выборок...")
train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length=512)
val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_length=512)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)
print("DataLoaders созданы.")

print("Инициализация оптимизатора...")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

def train(model, data_loader, optimizer):
    model.train()
    total_loss = 0
    print("Начало обучения на эпохе...")
    for step, batch in enumerate(data_loader):
        if step % 10 == 0:
            print(f"  Обработка батча {step + 1}/{len(data_loader)}...")

        inputs = {key: val.to(device) for key, val in batch.items()}

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print("Обучение на эпохе завершено.")
    return total_loss / len(data_loader)

def evaluate(model, data_loader):
    model.eval()
    preds, true_labels = []
    print("Начало оценки...")
    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            if step % 10 == 0:
                print(f"  Обработка батча {step + 1}/{len(data_loader)}...")

            inputs = {key: val.to(device) for key, val in batch.items()}
            
            outputs = model(**inputs)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(inputs['labels'].cpu().numpy())
    accuracy = accuracy_score(true_labels, preds)
    report = classification_report(true_labels, preds)
    print("Оценка завершена.")
    return accuracy, report

epochs = 3
print("Начало обучения модели...")
for epoch in range(epochs):
    print(f"\n=== Эпоха {epoch + 1}/{epochs} ===")
    train_loss = train(model, train_loader, optimizer)
    val_accuracy, val_report = evaluate(model, val_loader)
    print(f"Эпоха {epoch + 1} завершена.")
    print(f"Потери на обучении: {train_loss}")
    print(f"Точность на валидации: {val_accuracy}")
    print(f"Отчет по валидации:\n{val_report}")

print("Сохранение модели...")
model.save_pretrained('./text_classifier_model')
tokenizer.save_pretrained('./text_classifier_model')
print("Модель успешно сохранена!")
