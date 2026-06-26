import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
from collections import Counter

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from datetime import datetime

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_data = datasets.ImageFolder("dataset/train", transform=transform)
test_data = datasets.ImageFolder("dataset/test", transform=transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16)

print("Dataset train size:", len(train_data))
print("Dataset test size:", len(test_data))
print("Class mapping:", train_data.class_to_idx)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

class_counts = Counter(train_data.targets)
total = sum(class_counts.values())

weights = [total / class_counts[i] for i in range(len(class_counts))]
weights = torch.tensor(weights, dtype=torch.float32).to(device)

print("Class weights:", weights)

criterion = nn.CrossEntropyLoss(weight=weights)

for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=1
)

stage = 1

train_losses = []
accuracies = []

patience = 10
no_improve = 0
best_acc = 0

best_y_true = None
best_y_pred = None

for epoch in range(20):

    if epoch == 3 and stage == 1:
        print("\n🚀 Fine-tuning: unfreezing layer4")

        for param in model.layer4.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.0001
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=1
        )

        stage = 2

    model.train()
    total_loss = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 2 == 0:
            print(f"Epoch {epoch+1}, Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    model.eval()
    correct = 0
    total_samples = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total_samples += labels.size(0)
            correct += (predicted == labels).sum().item()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = 100 * correct / total_samples
    accuracies.append(acc)

    print(f"\n📊 Epoch {epoch+1}")
    print(f"Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=train_data.classes))

    scheduler.step(acc)

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "model.pth")

        best_y_true = y_true.copy()
        best_y_pred = y_pred.copy()

        print(f"💾 Saved best model (Accuracy: {acc:.2f}%)")
        no_improve = 0
    else:
        no_improve += 1
        print(f"⏳ No improvement: {no_improve}/{patience}")

    if no_improve >= patience:
        print("⛔ Early stopping triggered")
        break

plt.figure()
plt.plot(train_losses)
plt.title("Loss по епохах")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("loss.png")
plt.close()

plt.figure()
plt.plot(accuracies)
plt.title("Accuracy по епохах")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("accuracy.png")
plt.close()

print("📊 Графіки збережено")

cm = confusion_matrix(best_y_true, best_y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=train_data.classes)
disp.plot()
plt.savefig("confusion_matrix.png")
plt.close()

print("🧩 Confusion Matrix збережено")

if best_y_true is not None:
    report = classification_report(
        best_y_true,
        best_y_pred,
        target_names=train_data.classes
    )

    with open("report.txt", "w", encoding="utf-8") as f:
        f.write("Best Model Classification Report\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Accuracy: {best_acc:.2f}%\n\n")
        f.write(report)

    print("📄 Report збережено (report.txt)")